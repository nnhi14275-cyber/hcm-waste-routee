"""
app.py
------
Streamlit UI - Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt
bằng Google OR-Tools + Guided Local Search (GLS), dùng OSRM/OSM làm mạng
lưới đường thực tế.

PHẠM VI NGHIÊN CỨU: Tuyến Lê Văn Việt và khu vực lân cận, TP. Thủ Đức, TP.HCM.
"""

from __future__ import annotations

import io

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from baseline import nearest_neighbor_baseline
from data_generator import DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
from optimizer import OptimizeConfig, solve_cvrp
from routing import DEFAULT_OSRM_BASE_URL, OSRMError, get_osrm_matrices, get_osrm_route_geometry

st.set_page_config(page_title="Tối ưu tuyến thu gom rác - Lê Văn Việt", layout="wide")

# ============================================================================
# HEADER - Ghi rõ phạm vi & giả định nghiên cứu (bắt buộc theo yêu cầu đề tài)
# ============================================================================
st.title("Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt")

st.markdown(
    """
| Hạng mục | Nội dung |
|---|---|
| **Khu vực thử nghiệm** | Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ) |
| **Routing engine** | OpenStreetMap + OSRM |
| **Optimization** | Google OR-Tools + Guided Local Search |
| **Real-time traffic** | Chưa xét |
| **Baseline** | Simulated baseline – Nearest Neighbor / Greedy |
"""
)
st.caption(
    "Đây là mô hình nghiên cứu/prototype phục vụ mục đích học thuật (Green Logistics), "
    "KHÔNG phải hệ thống điều hành xe thu gom rác thực tế."
)
st.divider()

# ============================================================================
# SIDEBAR - CẤU HÌNH
# ============================================================================
with st.sidebar:
    st.header("1. Dữ liệu")
    data_mode = st.radio("Nguồn dữ liệu", ["Dữ liệu demo (Lê Văn Việt)", "Upload CSV/XLSX"])

    if data_mode == "Dữ liệu demo (Lê Văn Việt)":
        num_points = st.slider("Số điểm thu gom (demo, quanh Lê Văn Việt)", 20, 30, 25)
        seed = st.number_input("Random seed", value=42, step=1)
        use_tw_demo = st.checkbox("Sinh time window demo (VRPTW)", value=False)
        st.caption("Case study quy mô nhỏ: 20–30 điểm thu gom + 1 depot, phù hợp 2–3 xe.")
    else:
        uploaded_file = st.file_uploader("Upload file (CSV hoặc XLSX)", type=["csv", "xlsx"])
        st.caption(
            "Cột bắt buộc: node_id, latitude, longitude, waste_kg. "
            "Tuỳ chọn: service_time, time_window_start, time_window_end, is_depot."
        )

    st.header("2. Routing (OSRM)")
    osrm_base_url = st.text_input("OSRM base URL", value=DEFAULT_OSRM_BASE_URL)
    allow_fallback = st.checkbox(
        "Cho phép fallback Haversine nếu OSRM lỗi (KHÔNG khuyến nghị)", value=False
    )
    if allow_fallback:
        st.warning("Fallback mode – không sử dụng mạng lưới đường thực tế nếu được kích hoạt.")
        fallback_speed = st.slider("Tốc độ giả định cho fallback (km/h)", 10, 50, 25)
    else:
        fallback_speed = 25

    st.header("3. Xe & ràng buộc")
    num_vehicles = st.slider("Số xe tối đa (upper bound cho OR-Tools)", 1, 10, 3)
    st.caption("Case study quy mô nhỏ: mặc định 2-3 xe thu gom.")
    vehicle_capacity_kg = st.number_input("Vehicle capacity (kg)", value=1000, step=50)
    max_route_hours = st.slider("Max route duration (giờ)", 1.0, 8.0, 4.0, step=0.5)

    st.header("4. Thuật toán tối ưu (OR-Tools)")
    use_gls = st.checkbox("Bật Guided Local Search (GLS)", value=True)
    first_solution_strategy = st.selectbox(
        "Chiến lược khởi tạo (initial solution)",
        ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"],
    )
    time_limit_sec = st.slider("Thời gian chạy tối ưu (giây)", 5, 120, 20)

    st.header("5. Hệ số tiêu hao & phát thải (có thể chỉnh)")
    fuel_rate_l_per_km = st.number_input("Fuel rate (lít/km)", value=0.35, step=0.01, format="%.2f")
    emission_factor_kg_per_l = st.number_input(
        "Emission factor (kg CO2 / lít nhiên liệu)", value=2.68, step=0.01, format="%.2f"
    )

    run_btn = st.button("Chạy tối ưu", type="primary", use_container_width=True)


# ============================================================================
# LOAD DỮ LIỆU
# ============================================================================
def _load_data() -> pd.DataFrame | None:
    if data_mode == "Dữ liệu demo (Lê Văn Việt)":
        cfg = DemoConfig(num_points=num_points, seed=int(seed), use_time_windows=use_tw_demo)
        return generate_demo_data(cfg)

    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return load_points_from_dataframe(raw)


if "df_points" not in st.session_state:
    st.session_state.df_points = None
if "results" not in st.session_state:
    st.session_state.results = None

df_points = _load_data()
if df_points is not None:
    st.session_state.df_points = df_points

if st.session_state.df_points is None:
    st.info("Vui lòng upload dữ liệu hoặc dùng dữ liệu demo, sau đó nhấn **Chạy tối ưu**.")
    st.stop()

df_points = st.session_state.df_points

with st.expander("Xem dữ liệu điểm thu gom", expanded=False):
    st.dataframe(df_points, use_container_width=True)

# ============================================================================
# CHẠY PIPELINE KHI NHẤN NÚT
# ============================================================================
if run_btn:
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    demands = df_points["waste_kg"].tolist()
    service_times_s = (df_points["service_time"] * 60).tolist()
    time_windows_s = list(
        zip(df_points["time_window_start"] * 60, df_points["time_window_end"] * 60)
    )
    use_tw = df_points["time_window_start"].sum() > 0 or (
        data_mode == "Dữ liệu demo (Lê Văn Việt)" and use_tw_demo
    )

    with st.spinner("Đang gọi OSRM để lấy ma trận khoảng cách/thời gian đường bộ..."):
        try:
            matrix_result = get_osrm_matrices(
                coords,
                base_url=osrm_base_url,
                allow_haversine_fallback=allow_fallback,
                fallback_avg_speed_kmh=fallback_speed,
            )
        except OSRMError as exc:
            st.error(str(exc))
            st.stop()

    if matrix_result.source == "HAVERSINE_FALLBACK":
        st.warning(matrix_result.warning)
    else:
        st.success("Đã lấy ma trận khoảng cách/thời gian từ OSRM (Routing: OpenStreetMap + OSRM).")

    dist_m = matrix_result.distance_matrix_m
    dur_s = matrix_result.duration_matrix_s

    depot_index = int(df_points.index[df_points["is_depot"]][0])

    # ---------------- BASELINE ----------------
    with st.spinner("Đang xây dựng baseline (Nearest Neighbor / Greedy)..."):
        try:
            baseline_routes = nearest_neighbor_baseline(
                dist_m, dur_s, demands, service_times_s,
                vehicle_capacity_kg=vehicle_capacity_kg,
                depot_index=depot_index,
                max_route_time_s=max_route_hours * 3600,
                max_vehicles=max(num_vehicles, 20),
            )
        except RuntimeError as exc:
            st.error(f"Baseline thất bại: {exc}")
            st.stop()

    # ---------------- OPTIMIZED (OR-TOOLS + GLS) ----------------
    with st.spinner("Đang chạy OR-Tools + Guided Local Search..."):
        opt_config = OptimizeConfig(
            num_vehicles=max(num_vehicles, len(baseline_routes)),
            vehicle_capacity_kg=vehicle_capacity_kg,
            depot_index=depot_index,
            use_gls=use_gls,
            first_solution_strategy=first_solution_strategy,
            time_limit_sec=time_limit_sec,
            max_route_time_s=max_route_hours * 3600,
            use_time_windows=use_tw,
            time_windows_s=time_windows_s,
        )
        optimized_routes, solved, msg = solve_cvrp(dist_m, dur_s, demands, service_times_s, opt_config)

    if not solved:
        st.error(msg)
        st.stop()

    st.session_state.results = {
        "df_points": df_points,
        "coords": coords,
        "matrix_result": matrix_result,
        "baseline_routes": baseline_routes,
        "optimized_routes": optimized_routes,
        "depot_index": depot_index,
        "osrm_base_url": osrm_base_url,
        "fuel_rate_l_per_km": fuel_rate_l_per_km,
        "emission_factor_kg_per_l": emission_factor_kg_per_l,
    }

# ============================================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================================
if st.session_state.results is None:
    st.stop()

res = st.session_state.results
df_points = res["df_points"]
coords = res["coords"]
baseline_routes = res["baseline_routes"]
optimized_routes = res["optimized_routes"]
depot_index = res["depot_index"]
fuel_rate = res["fuel_rate_l_per_km"]
emission_factor = res["emission_factor_kg_per_l"]


def _aggregate(routes) -> dict:
    total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0
    travel_time_min = sum(r.travel_time_s for r in routes) / 60.0
    service_time_min = sum(r.service_time_s for r in routes) / 60.0
    total_time_min = sum(r.total_route_time_s for r in routes) / 60.0
    num_vehicles_used = len(routes)
    total_waste_kg = sum(r.collected_waste_kg for r in routes)
    avg_util = (
        sum(r.capacity_utilization_pct for r in routes) / len(routes) if routes else 0.0
    )
    fuel_l = total_distance_km * fuel_rate
    co2_kg = fuel_l * emission_factor
    return {
        "Total distance (km)": round(total_distance_km, 2),
        "Travel time (min)": round(travel_time_min, 1),
        "Service time (min)": round(service_time_min, 1),
        "Total route time (min)": round(total_time_min, 1),
        "Number of vehicles": num_vehicles_used,
        "Total waste collected (kg)": round(total_waste_kg, 1),
        "Average capacity utilization (%)": round(avg_util, 1),
        "Estimated fuel consumption (L)": round(fuel_l, 2),
        "Estimated CO2 emissions (kg)": round(co2_kg, 2),
    }


baseline_kpi = _aggregate(baseline_routes)
optimized_kpi = _aggregate(optimized_routes)


def _reduction(base, opt):
    if base == 0:
        return 0.0
    return round((base - opt) / base * 100, 1)


kpi_rows = []
for key in baseline_kpi:
    row = {"Chỉ tiêu": key, "Baseline": baseline_kpi[key], "Optimized": optimized_kpi[key]}
    if key in (
        "Total distance (km)", "Total route time (min)",
        "Estimated CO2 emissions (kg)", "Estimated fuel consumption (L)",
    ):
        row["Reduction (%)"] = _reduction(baseline_kpi[key], optimized_kpi[key])
    else:
        row["Reduction (%)"] = "-"
    kpi_rows.append(row)

st.subheader("So sánh KPI: Baseline vs Optimized")
st.caption(
    "Baseline = tuyến cơ sở mô phỏng bằng heuristic Nearest Neighbor/Greedy "
    f"(nguồn khoảng cách/thời gian: {res['matrix_result'].source}). "
    "CO2 emissions là giá trị ước tính (Estimated CO2 emissions), không phải đo trực tiếp."
)
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

# ---------------- Chi tiết từng tuyến ----------------
node_ids = df_points["node_id"].tolist()


def _route_label(route) -> str:
    names = [node_ids[i] for i in route.node_sequence]
    return " → ".join(names)


col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline routes**")
    for r in baseline_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )
with col_b:
    st.markdown("**Optimized routes (OR-Tools + GLS)**")
    for r in optimized_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )

st.divider()

# ============================================================================
# BẢN ĐỒ (Folium + OSRM road geometry thực tế)
# ============================================================================
st.subheader("Bản đồ tuyến (road geometry thực tế từ OSRM)")

center_lat, center_lon = DEPOT_LOCATION
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# Depot & các điểm thu gom
for idx, row in df_points.iterrows():
    if row["is_depot"]:
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup="DEPOT",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(fmap)
    else:
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=5,
            popup=f"{row['node_id']} - {row['waste_kg']:.0f} kg",
            color="#555555",
            fill=True,
            fill_opacity=0.8,
        ).add_to(fmap)


def _draw_routes(routes, color, label_prefix):
    for r in routes:
        ordered_coords = tuple(coords[i] for i in r.node_sequence)
        geometry = get_osrm_route_geometry(ordered_coords, base_url=res["osrm_base_url"])
        if not geometry:
            # OSRM route service không khả dụng cho tuyến này -> vẽ tạm bằng
            # đường nối các điểm (KHÔNG phải road geometry thực tế), có ghi chú.
            geometry = list(ordered_coords)
            dash = "5, 10"
        else:
            dash = None
        folium.PolyLine(
            geometry,
            color=color,
            weight=4,
            opacity=0.8,
            dash_array=dash,
            tooltip=f"{label_prefix} - Vehicle {r.vehicle_id} ({r.total_distance_m/1000:.2f} km)",
        ).add_to(fmap)


_draw_routes(baseline_routes, "#1f77b4", "Baseline")
_draw_routes(optimized_routes, "#d62728", "Optimized")

st.caption(
    "🔵 Xanh dương = Baseline (Nearest Neighbor/Greedy) · 🔴 Đỏ = Optimized (OR-Tools + GLS). "
    "Nét đứt (nếu có) nghĩa là OSRM Route Service không trả về được geometry cho đoạn đó."
)
st_folium(fmap, use_container_width=True, height=600, returned_objects=[])
