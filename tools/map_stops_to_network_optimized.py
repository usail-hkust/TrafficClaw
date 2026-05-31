#!/usr/bin/env python3
"""
将SUMO busStop映射到路网的lane
使用经纬度坐标查找最近的道路

性能优化版本：只读取路网一次
"""

import xml.etree.ElementTree as ET
import sys
import json
from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

ROUTABLE_CLASSES = (
    "passenger",
    "bus",
    "tram",
    "rail_urban",
    "rail_electric",
    "rail",
    "rail_fast",
    "ship",
    "custom1",
    "custom2",
)


def parse_stops_file(stops_xml):
    """解析站点XML文件，提取站点信息"""
    tree = ET.parse(stops_xml)
    root = tree.getroot()

    stops = []
    for bus_stop in root.findall('busStop'):
        stop_id = bus_stop.get('id')
        name = bus_stop.get('name', '')

        # 获取地理坐标
        geo_param = bus_stop.find("param[@key='geoPosition']")
        if geo_param is not None:
            lon, lat = map(float, geo_param.get('value').split(','))
            stops.append({
                'id': stop_id,
                'name': name,
                'lon': lon,
                'lat': lat
            })

    return stops


def _is_valid_lane(lane, require_connectivity=True, require_vclasses=None):
    """Filter out internal/footpath lanes that break routing."""
    edge = lane.getEdge()
    func = edge.getFunction()

    if edge.getID().startswith(':'):
        return False

    invalid_funcs = {"internal", "connector", "crossing", "walkingarea"}
    if func in invalid_funcs:
        return False

    try:
        allowed_classes = require_vclasses or ROUTABLE_CLASSES
        if not any(lane.allows(cls) for cls in allowed_classes):
            return False
    except Exception:
        return False

    if require_connectivity:
        if not edge.getOutgoing() or not edge.getIncoming():
            return False

    return True


def _lane_heading_rad(lane) -> Optional[float]:
    try:
        shape = lane.getShape()
        if not shape or len(shape) < 2:
            return None
        (x0, y0) = shape[0]
        (x1, y1) = shape[-1]
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) + abs(dy) < 1e-9:
            return None
        return math.atan2(dy, dx)
    except Exception:
        return None


def _angle_diff_rad(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def _stop_numeric_id(stop_id: str) -> Optional[str]:
    if not stop_id:
        return None
    s = str(stop_id).strip()
    if s.startswith("bus_stop_"):
        return s[len("bus_stop_") :]
    if s.startswith("subway_stop_"):
        return s[len("subway_stop_") :]
    if s.startswith("rail_stop_"):
        return s[len("rail_stop_") :]
    if s.startswith("ferry_stop_"):
        return s[len("ferry_stop_") :]
    return None


def _build_preferred_headings(
    *,
    timetable_json: str,
    stops: List[dict],
    net,
    stop_prefix: str,
) -> Dict[str, float]:
    """
    Build a preferred travel heading for each stop based on timetable stop sequences.
    Returns mapping from SUMO stop id (e.g. bus_stop_303345) -> heading angle (rad).
    """
    stop_xy: Dict[str, Tuple[float, float]] = {}
    for s in stops:
        sid = s.get("id")
        if not sid:
            continue
        try:
            x, y = net.convertLonLat2XY(float(s["lon"]), float(s["lat"]))
            stop_xy[str(sid)] = (x, y)
        except Exception:
            continue

    with open(timetable_json, "r") as f:
        data = json.load(f)

    accum: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    counts: Dict[str, int] = defaultdict(int)

    for _route_id, route_data in data.items():
        routes_info = (route_data or {}).get("routes") or {}
        if not isinstance(routes_info, dict):
            continue
        for _pattern, pat in routes_info.items():
            seq = (pat or {}).get("stops") or []
            if not isinstance(seq, list) or len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                a = f"{stop_prefix}{seq[i]}"
                b = f"{stop_prefix}{seq[i + 1]}"
                if a not in stop_xy or b not in stop_xy:
                    continue
                (xa, ya) = stop_xy[a]
                (xb, yb) = stop_xy[b]
                dx = xb - xa
                dy = yb - ya
                if abs(dx) + abs(dy) < 1e-6:
                    continue
                ang = math.atan2(dy, dx)
                sx, sy = accum[a]
                accum[a] = (sx + math.cos(ang), sy + math.sin(ang))
                counts[a] += 1

    out: Dict[str, float] = {}
    for sid, (sx, sy) in accum.items():
        n = counts.get(sid, 0)
        if n <= 0:
            continue
        mag = math.hypot(sx, sy) / float(n)
        # If headings cancel out (stop used in opposite directions), don't enforce.
        if mag < 0.25:
            continue
        out[sid] = math.atan2(sy, sx)
    return out


def find_nearest_edge(
    net,
    lon,
    lat,
    *,
    require_vclasses=None,
    max_radius=4000,
    preferred_heading_rad: Optional[float] = None,
    max_heading_diff_deg: float = 90.0,
):
    """
    使用已加载的路网对象找到最近的edge（优化版）

    Args:
        net: 已加载的sumolib.net对象
        lon: 经度
        lat: 纬度

    Returns:
        edge_id, lane_id, pos
    """
    try:
        # 将经纬度转换为SUMO坐标
        x, y = net.convertLonLat2XY(lon, lat)

        # 查找最近的lane
        radius_steps = [100, 500, 1000, 2000, 4000]
        if isinstance(max_radius, (int, float)) and max_radius > 0:
            radius_steps = [r for r in radius_steps if r <= max_radius]
            if not radius_steps:
                radius_steps = [int(max_radius)]

        lanes = []
        for radius in radius_steps:
            lanes = net.getNeighboringLanes(x, y, radius)
            if lanes:
                break

        if lanes:
            max_heading_diff = math.radians(float(max_heading_diff_deg))

            def pick_lane(candidates, require_connectivity):
                scored = []
                for lane, dist in candidates:
                    if not _is_valid_lane(
                        lane,
                        require_connectivity=require_connectivity,
                        require_vclasses=require_vclasses,
                    ):
                        continue
                    if preferred_heading_rad is None:
                        scored.append((float(dist), 0.0, lane))
                        continue
                    h = _lane_heading_rad(lane)
                    if h is None:
                        continue
                    d_ang = _angle_diff_rad(h, preferred_heading_rad)
                    if d_ang > max_heading_diff:
                        continue
                    scored.append((float(dist), float(d_ang), lane))
                if not scored:
                    return None, None
                # Prefer direction match first (angle), then distance.
                scored.sort(key=lambda t: (t[1], t[0]))
                dist, _dang, lane = scored[0]
                return lane, dist

            valid_lane, distance = pick_lane(lanes, True)

            if valid_lane is None:
                # 仍未找到：放宽连通性要求（但仍要求 vClass 许可）
                for radius in reversed(radius_steps):
                    lanes = net.getNeighboringLanes(x, y, radius)
                    valid_lane, distance = pick_lane(lanes, False)
                    if valid_lane:
                        break

            if valid_lane is None:
                # If direction constraint blocked everything, fall back to nearest allowed lane (ignore heading).
                def pick_lane_ignore_heading(candidates, require_connectivity):
                    scored = []
                    for lane, dist in candidates:
                        if not _is_valid_lane(
                            lane,
                            require_connectivity=require_connectivity,
                            require_vclasses=require_vclasses,
                        ):
                            continue
                        scored.append((float(dist), lane))
                    if not scored:
                        return None, None
                    scored.sort(key=lambda t: t[0])
                    dist, lane = scored[0]
                    return lane, dist

                valid_lane, distance = pick_lane_ignore_heading(lanes, False)
                if valid_lane is None:
                    return None, None, None, None

            # 计算在lane上的位置
            pos = valid_lane.getClosestLanePosAndDist((x, y))[0]

            return valid_lane.getEdge().getID(), valid_lane.getID(), pos, distance
        return None, None, None, None

    except Exception as e:
        print(f"查找最近edge失败: {e}", file=sys.stderr)
        return None, None, None, None


def map_stops_to_network(
    stops_xml,
    net_file,
    output_xml,
    *,
    require_vclasses=None,
    max_radius=4000,
    timetable_json: Optional[str] = None,
    stop_prefix: str = "bus_stop_",
    max_heading_diff_deg: float = 90.0,
):
    """
    将站点映射到路网（优化版：只读取路网一次）

    Args:
        stops_xml: 输入的站点XML（含PLACEHOLDER）
        net_file: SUMO路网文件
        output_xml: 输出的映射后站点XML
    """
    print(f"读取站点文件: {stops_xml}")
    stops = parse_stops_file(stops_xml)
    print(f"找到 {len(stops)} 个站点")

    print(f"\n读取路网文件: {net_file}")
    print("  ⏳ 正在加载路网（4.3GB，仅此一次）...")

    try:
        import sumolib
    except ImportError:
        print("错误: 需要安装sumolib")
        print("请运行: pip install sumolib")
        sys.exit(1)

    # ⭐ 关键优化：只读取路网一次
    net = sumolib.net.readNet(net_file)
    print(f"  ✓ 路网加载完成")
    print(f"    - Edges: {len(net.getEdges()):,}")
    print(f"    - Junctions: {len(net.getNodes()):,}")

    # 创建输出XML
    root = ET.Element("additional")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/additional_file.xsd")

    mapped_count = 0
    unmapped_stops = []

    preferred_headings: Dict[str, float] = {}
    if timetable_json:
        print(f"\n构建站点方向偏好（来自时刻表站序）: {timetable_json}")
        preferred_headings = _build_preferred_headings(
            timetable_json=timetable_json,
            stops=stops,
            net=net,
            stop_prefix=stop_prefix,
        )
        print(f"  ✓ 已生成 {len(preferred_headings)} 个站点的方向偏好")

    print(f"\n开始映射站点到路网...")
    for i, stop in enumerate(stops, 1):
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(stops)} 个站点 ({100*i/len(stops):.1f}%)")

        # ⭐ 关键优化：传递已加载的路网对象，而不是文件路径
        pref = preferred_headings.get(str(stop.get("id") or ""))
        edge_id, lane_id, pos, distance = find_nearest_edge(
            net,
            stop['lon'],
            stop['lat'],
            require_vclasses=require_vclasses,
            max_radius=max_radius,
            preferred_heading_rad=pref,
            max_heading_diff_deg=max_heading_diff_deg,
        )

        if lane_id:
            # 创建busStop元素
            bus_stop = ET.SubElement(root, "busStop")
            bus_stop.set("id", stop['id'])
            bus_stop.set("name", stop['name'])
            lane_obj = net.getLane(lane_id)
            lane_len = lane_obj.getLength() if lane_obj is not None else max(pos, 0) + 10
            start_pos = max(0.0, min(pos - 10, lane_len))
            end_pos = min(lane_len, max(pos + 10, start_pos + 0.1))

            bus_stop.set("lane", lane_id)
            bus_stop.set("startPos", f"{start_pos:.2f}")
            bus_stop.set("endPos", f"{end_pos:.2f}")

            # 保留原始坐标信息
            param_geo = ET.SubElement(bus_stop, "param")
            param_geo.set("key", "geoPosition")
            param_geo.set("value", f"{stop['lon']},{stop['lat']}")

            param_edge = ET.SubElement(bus_stop, "param")
            param_edge.set("key", "edge")
            param_edge.set("value", edge_id)

            param_distance = ET.SubElement(bus_stop, "param")
            param_distance.set("key", "distanceToLane")
            param_distance.set("value", f"{distance:.2f}")

            mapped_count += 1
        else:
            unmapped_stops.append(stop)
            print(f"  ⚠ 无法映射站点: {stop['name']} ({stop['lon']}, {stop['lat']})")

    # 写入文件
    print(f"\n写入输出文件...")
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    with open(output_xml, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

    print(f"\n{'='*60}")
    print(f"✓ 站点映射完成")
    print(f"  成功映射: {mapped_count}/{len(stops)} 个站点 ({100*mapped_count/len(stops):.1f}%)")
    print(f"  未映射: {len(unmapped_stops)} 个站点")
    print(f"  输出文件: {output_xml}")
    print(f"{'='*60}")

    # 保存未映射站点列表
    if unmapped_stops:
        unmapped_file = output_xml.replace('.add.xml', '_unmapped.json')
        with open(unmapped_file, 'w') as f:
            json.dump(unmapped_stops, f, indent=2, ensure_ascii=False)
        print(f"  未映射站点保存至: {unmapped_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='将SUMO busStop映射到路网（性能优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python map_stops_to_network_optimized.py \\
      -s sumo_output/subway_stops.add.xml \\
      -n sumo_network/newyork.net.xml \\
      -o sumo_network/subway_stops_mapped.add.xml
        """
    )

    parser.add_argument('-s', '--stops', required=True,
                        help='输入站点XML文件')
    parser.add_argument('-n', '--network', required=True,
                        help='SUMO路网文件')
    parser.add_argument('-o', '--output', required=True,
                        help='输出映射后的站点XML文件')
    parser.add_argument(
        '--require-vclass',
        action='append',
        default=[],
        help='只映射到允许该vClass的lane（可重复，例如 --require-vclass bus）',
    )
    parser.add_argument(
        '--max-radius',
        type=int,
        default=4000,
        help='最大搜索半径（米，默认4000）',
    )
    parser.add_argument('--timetable', help='可选：时刻表JSON（用于按站序推断行进方向，提升映射一致性）')
    parser.add_argument('--stop-prefix', default='bus_stop_', help='站点ID前缀（默认 bus_stop_）')
    parser.add_argument(
        '--max-heading-diff-deg',
        type=float,
        default=90.0,
        help='方向约束：候选lane与站序方向最大允许偏差(度，默认90)',
    )

    args = parser.parse_args()

    require_vclasses = [x.strip() for x in (args.require_vclass or []) if str(x).strip()]
    map_stops_to_network(
        args.stops,
        args.network,
        args.output,
        require_vclasses=require_vclasses or None,
        max_radius=int(args.max_radius),
        timetable_json=args.timetable,
        stop_prefix=str(args.stop_prefix),
        max_heading_diff_deg=float(args.max_heading_diff_deg),
    )


if __name__ == '__main__':
    main()
