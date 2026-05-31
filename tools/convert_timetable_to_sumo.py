#!/usr/bin/env python3
"""
将时刻表JSON转换为SUMO vehicle/flow定义
SUMO Documentation: https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import os
from datetime import datetime, timedelta
from collections import defaultdict
import csv
from typing import Dict, Optional, Tuple


def time_to_seconds(time_str):
    """
    将时间字符串 (HH:MM:SS) 转换为秒数

    Args:
        time_str: 时间字符串，格式 "HH:MM:SS"

    Returns:
        int: 从午夜开始的秒数
    """
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0


def _normalize_service_date(s: str) -> str:
    return (s or "").strip().replace("-", "/")


def _count_service_dates(service_key: str) -> int:
    if not service_key:
        return 0
    return len([x for x in str(service_key).split("|") if x.strip()])


def _select_single_service_group(
    services_info: Dict[str, dict],
    *,
    service_key: Optional[str],
    service_date: Optional[str],
    service_policy: str,
) -> Tuple[Optional[str], Dict[str, dict]]:
    if not services_info:
        return None, {}

    if service_key:
        if service_key not in services_info:
            raise ValueError(f"未找到指定的service key: {service_key}")
        return service_key, {service_key: services_info[service_key]}

    if service_date:
        normalized = _normalize_service_date(service_date)
        for k in services_info.keys():
            if normalized and normalized in str(k):
                return str(k), {str(k): services_info[k]}

    if service_policy == "all":
        return None, services_info

    keys = list(services_info.keys())
    if service_policy == "first":
        k = str(keys[0])
        return k, {k: services_info[keys[0]]}
    if service_policy == "largest":
        best = max(keys, key=lambda x: _count_service_dates(str(x)))
        k = str(best)
        return k, {k: services_info[best]}
    if service_policy == "most_trips":
        best = max(
            keys,
            key=lambda x: len((services_info.get(x) or {}).get("trips") or {}),
        )
        k = str(best)
        return k, {k: services_info[best]}

    raise ValueError(f"未知service policy: {service_policy}")


def create_routes_and_vehicles(
    timetable_file,
    stops_mapping_file,
    output_file,
    transport_mode="bus",
    max_vehicles=None,
    max_vehicles_per_line: Optional[int] = None,
    depart_begin: Optional[int] = None,
    depart_end: Optional[int] = None,
    service_policy: str = "largest",
    service_date: Optional[str] = None,
    service_key: Optional[str] = None,
):
    """
    将时刻表JSON转换为SUMO routes和vehicles XML

    Args:
        timetable_file: 时刻表JSON文件路径
        stops_mapping_file: 站点映射CSV文件路径
        output_file: 输出XML文件路径
        transport_mode: 交通模式
        max_vehicles: 最大车辆数量限制（用于测试）
    """

    # 读取时刻表数据
    print(f"读取时刻表: {timetable_file}")
    with open(timetable_file, 'r') as f:
        timetable_data = json.load(f)

    def load_stop_mapping_csv(path: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return mapping
            if 'stop_id' not in reader.fieldnames or 'sumo_stop_id' not in reader.fieldnames:
                raise ValueError(f"站点映射CSV缺少必要列 stop_id/sumo_stop_id: {reader.fieldnames}")
            for row in reader:
                sid = str(row.get('stop_id', '')).strip()
                ssumo = str(row.get('sumo_stop_id', '')).strip()
                if sid and ssumo:
                    mapping[sid] = ssumo
        return mapping

    # 读取站点映射（避免依赖pandas）
    stop_id_to_sumo = load_stop_mapping_csv(stops_mapping_file)

    print(f"加载 {len(stop_id_to_sumo)} 个站点映射")

    # 创建XML根元素
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    # 添加注释
    comment = ET.Comment(f"""
    SUMO Routes and Vehicles converted from timetable data
    Transport Mode: {transport_mode}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    root.insert(0, comment)

    # 定义车辆类型
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", f"{transport_mode}_default")

    # 根据交通模式设置vClass
    vclass_mapping = {
        'bus': 'bus',
        'subway': 'rail_urban',
        'rail': 'rail',
        'ferry': 'ship'
    }
    vtype.set("vClass", vclass_mapping.get(transport_mode, 'bus'))

    # 设置车辆参数
    if transport_mode == 'bus':
        vtype.set("length", "12.0")
        vtype.set("maxSpeed", "20.0")  # ~72 km/h
        vtype.set("accel", "1.5")
        vtype.set("decel", "2.0")
        vtype.set("color", "yellow")
    elif transport_mode == 'subway':
        vtype.set("length", "100.0")
        vtype.set("maxSpeed", "30.0")  # ~108 km/h
        vtype.set("color", "blue")
    elif transport_mode == 'rail':
        vtype.set("length", "150.0")
        vtype.set("maxSpeed", "40.0")  # ~144 km/h
        vtype.set("color", "red")
    elif transport_mode == 'ferry':
        vtype.set("length", "50.0")
        vtype.set("maxSpeed", "10.0")  # ~36 km/h
        vtype.set("color", "cyan")

    # 统计信息
    total_routes = 0
    total_vehicles = 0
    route_vehicle_count = defaultdict(int)
    vehicles_per_line = defaultdict(int)

    # 遍历每条线路
    for route_id, route_data in timetable_data.items():
        print(f"\n处理线路: {route_id}")

        routes_info = route_data.get('routes', {})
        services_info = route_data.get('services', {})

        selected_service_key, selected_services_info = _select_single_service_group(
            services_info,
            service_key=service_key,
            service_date=service_date,
            service_policy=service_policy,
        )
        if service_policy != "all" or service_date or service_key:
            if selected_service_key:
                dates_n = _count_service_dates(selected_service_key)
                trips_n = len((selected_services_info.get(selected_service_key) or {}).get("trips") or {})
                first_date = str(selected_service_key).split("|", 1)[0]
                print(f"  选择service组: first={first_date} dates={dates_n} trips={trips_n} (policy={service_policy})")
            else:
                print(f"  选择service组: (none) (policy={service_policy})")
        else:
            print(f"  使用全部service组: {len(services_info)}")

        # 为每个路线模式创建route定义
        for route_pattern, pattern_data in routes_info.items():
            stops_sequence = pattern_data['stops']
            trip_ids = pattern_data['trip_ids']
            trip_id_set = set(str(tid) for tid in trip_ids)

            # 创建SUMO route ID
            sumo_route_id = f"{transport_mode}_{route_id}_{route_pattern.replace('->', '_')}"
            # SUMO treats schedules as belonging to a 'line' with a fixed stop sequence.
            # A single GTFS route_id often has multiple patterns (e.g., directions/short turns),
            # so we separate SUMO line ids by pattern to avoid "ignoring schedule" warnings.
            sumo_line_id = sumo_route_id

            # 转换站点ID到SUMO格式
            sumo_stops = []
            for stop_id in stops_sequence:
                stop_id_str = str(stop_id)
                if stop_id_str in stop_id_to_sumo:
                    sumo_stops.append(stop_id_to_sumo[stop_id_str])
                else:
                    # 如果找不到映射，使用原始ID
                    sumo_stops.append(f"{transport_mode}_stop_{stop_id}")

            # 创建route元素（暂时使用占位符edges）；如果后续没有任何vehicle落在筛选窗口内，会移除
            route = ET.SubElement(root, "route")
            route.set("id", sumo_route_id)
            route.set("edges", "PLACEHOLDER")  # 需要在路网生成后填充/或用站点序列法生成
            route.set("color", "yellow" if transport_mode == 'bus' else "blue")

            param_stops = ET.SubElement(route, "param")
            param_stops.set("key", "stops_sequence")
            param_stops.set("value", ",".join(sumo_stops))

            created_any_vehicle_for_route = False

            # 遍历所有服务日期和trip，创建vehicle定义
            reached_line_cap = False
            for service_dates, service_data in selected_services_info.items():
                if max_vehicles_per_line and vehicles_per_line[route_id] >= max_vehicles_per_line:
                    reached_line_cap = True
                    break

                trips = (service_data or {}).get('trips', {})
                if not isinstance(trips, dict):
                    continue

                for trip_id_str, stops_times in trips.items():
                    trip_id = trip_id_str
                    if str(trip_id_str) not in trip_id_set:
                        continue

                    # 检查是否达到车辆数量限制
                    if max_vehicles and total_vehicles >= max_vehicles:
                        print(f"⚠ 达到车辆数量限制 ({max_vehicles})，停止生成")
                        break

                    if max_vehicles_per_line and vehicles_per_line[route_id] >= max_vehicles_per_line:
                        reached_line_cap = True
                        break

                    # 创建vehicle元素
                    vehicle = ET.SubElement(root, "vehicle")
                    vehicle_id = f"{transport_mode}_{route_id}_trip_{trip_id}"
                    vehicle.set("id", vehicle_id)
                    vehicle.set("type", f"{transport_mode}_default")
                    vehicle.set("route", sumo_route_id)
                    vehicle.set("line", sumo_line_id)

                    # 设置发车时间（使用第一个站点的到达时间）
                    if stops_times:
                        first_stop_time = stops_times[0]['arrive_time']
                        depart_seconds = time_to_seconds(first_stop_time)
                        if depart_begin is not None and depart_seconds < depart_begin:
                            # remove the vehicle element we just created
                            root.remove(vehicle)
                            continue
                        if depart_end is not None and depart_seconds > depart_end:
                            root.remove(vehicle)
                            continue
                        vehicle.set("depart", str(depart_seconds))

                        # 为每个站点添加stop元素
                        for stop_time_info in stops_times:
                            stop_id_str = str(stop_time_info['stop_id'])
                            arrive_time = stop_time_info['arrive_time']
                            until_seconds = time_to_seconds(arrive_time)

                            # 获取SUMO stop ID
                            if stop_id_str in stop_id_to_sumo:
                                sumo_stop_id = stop_id_to_sumo[stop_id_str]
                            else:
                                sumo_stop_id = f"{transport_mode}_stop_{stop_id_str}"

                            # 创建stop元素
                            stop = ET.SubElement(vehicle, "stop")
                            stop.set("busStop", sumo_stop_id)
                            stop.set("until", str(until_seconds))
                            stop.set("duration", "30")  # 默认停靠30秒

                    total_vehicles += 1
                    route_vehicle_count[sumo_route_id] += 1
                    vehicles_per_line[route_id] += 1
                    created_any_vehicle_for_route = True

                    if total_vehicles % 100 == 0:
                        print(f"  已生成 {total_vehicles} 个车辆...")

                if max_vehicles and total_vehicles >= max_vehicles:
                    break
                if reached_line_cap:
                    break

            if max_vehicles and total_vehicles >= max_vehicles:
                break

            if not created_any_vehicle_for_route:
                # remove route definition if no vehicles survived filtering
                root.remove(route)
            else:
                total_routes += 1

        if max_vehicles and total_vehicles >= max_vehicles:
            break

    # 使用ElementTree直接写入（带美化）
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)

    # 写入文件
    with open(output_file, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

    print(f"\n{'='*60}")
    print(f"✓ 成功生成 {output_file}")
    print(f"  总路线数: {total_routes}")
    print(f"  总车辆数: {total_vehicles}")
    print(f"{'='*60}")

    # 生成统计报告
    stats_file = output_file.replace('.rou.xml', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"SUMO Routes/Vehicles Generation Statistics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Transport Mode: {transport_mode}\n")
        f.write(f"Total Routes: {total_routes}\n")
        f.write(f"Total Vehicles: {total_vehicles}\n")
        f.write(f"\nVehicles per Route:\n")
        for route_id, count in sorted(route_vehicle_count.items()):
            f.write(f"  {route_id}: {count}\n")

    print(f"✓ 统计信息保存至: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description='将时刻表JSON转换为SUMO routes和vehicles XML格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换公交时刻表
  python convert_timetable_to_sumo.py \\
      -t Data/timetables/ferry_timetable.json \\
      -s bus_stops_route_mapping.csv \\
      -o bus_routes.rou.xml \\
      -m bus

  # 限制车辆数量（用于测试）
  python convert_timetable_to_sumo.py \\
      -t Data/timetables/ferry_timetable.json \\
      -s bus_stops_route_mapping.csv \\
      -o bus_routes_test.rou.xml \\
      -m bus \\
      --max-vehicles 100

  # 批量转换所有交通模式
  python convert_timetable_to_sumo.py --batch
        """
    )

    parser.add_argument('-t', '--timetable', help='时刻表JSON文件路径')
    parser.add_argument('-s', '--stops-mapping', help='站点映射CSV文件路径')
    parser.add_argument('-o', '--output', help='输出XML文件路径')
    parser.add_argument('-m', '--mode',
                        choices=['bus', 'subway', 'rail', 'ferry'],
                        help='交通模式')
    parser.add_argument('--max-vehicles', type=int,
                        help='最大车辆数量限制（用于测试）')
    parser.add_argument('--max-vehicles-per-line', type=int,
                        help='每条线路最多生成的车辆数量（避免只生成第一条线路）')
    parser.add_argument('--depart-begin', type=int,
                        help='只生成发车时间>=该值(秒)的车辆')
    parser.add_argument('--depart-end', type=int,
                        help='只生成发车时间<=该值(秒)的车辆')
    parser.add_argument(
        '--service-policy',
        choices=['all', 'first', 'largest', 'most_trips'],
        default='largest',
        help='每条线路选择哪个service组(all=全部; first=第一个; largest=日期数最多; most_trips=trip数最多)',
    )
    parser.add_argument('--service-date', help='优先选择包含该日期的service组(YYYY-MM-DD或YYYY/MM/DD)')
    parser.add_argument('--service-key', help='指定service组key(精确匹配；会覆盖service-policy/date)')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理所有交通模式')

    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        modes = ['bus', 'subway', 'rail', 'ferry']

        for mode in modes:
            timetable_file = f"Data/timetables/{mode}_timetable.json" if mode != 'bus' \
                else "Data/timetables/merged_bus_timetable.json"
            stops_mapping = f"{mode}_stops_route_mapping.csv"
            output_file = f"{mode}_routes.rou.xml"

            if os.path.exists(timetable_file) and os.path.exists(stops_mapping):
                print(f"\n{'='*60}")
                print(f"处理 {mode.upper()} 时刻表...")
                print(f"{'='*60}")
                create_routes_and_vehicles(
                    timetable_file,
                    stops_mapping,
                    output_file,
                    mode,
                    max_vehicles=args.max_vehicles,
                    max_vehicles_per_line=args.max_vehicles_per_line,
                    depart_begin=args.depart_begin,
                    depart_end=args.depart_end,
                    service_policy=args.service_policy,
                    service_date=args.service_date,
                    service_key=args.service_key,
                )
            else:
                print(f"⚠ 跳过 {mode}: 缺少必要文件")
                if not os.path.exists(timetable_file):
                    print(f"  缺少: {timetable_file}")
                if not os.path.exists(stops_mapping):
                    print(f"  缺少: {stops_mapping}")

    elif args.timetable and args.stops_mapping and args.output and args.mode:
        # 单个文件处理模式
        create_routes_and_vehicles(
            args.timetable,
            args.stops_mapping,
            args.output,
            args.mode,
            max_vehicles=args.max_vehicles,
            max_vehicles_per_line=args.max_vehicles_per_line,
            depart_begin=args.depart_begin,
            depart_end=args.depart_end,
            service_policy=args.service_policy,
            service_date=args.service_date,
            service_key=args.service_key,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
