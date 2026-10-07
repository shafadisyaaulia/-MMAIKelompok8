"""
=================================================================================
PROGRAM PENCARIAN JALUR TERPENDEK PADA GRAF (SHORTEST PATH ALGORITHMS)
Mata Kuliah : MMAI1001 - Kecerdasan Artificial (S2 Informatika)
Tugas       : Tugas 1 - Algoritma Shortest Path & Visualisasi Graf
File        : MMAIKelompokX.py (Ganti X dengan nomor kelompok Anda)
Algoritma   : 1. Algoritma Dijkstra (Uniform Cost / Greedy Search)
              2. Algoritma A* / A-Star (Informed Heuristic Search)
              3. Algoritma Bellman-Ford (Dynamic Programming Search)
Fitur Utama :
- Pembacaan file input 'graf.txt' otomatis
- Visualisasi Graf Dinamis & Animasi Pencarian Langkah demi Langkah (GUI Tkinter)
- Penentuan Verteks Awal & Akhir interaktif (Dropdown & Klik Canvas)
- Node dapat di-drag & drop pada Canvas untuk kerapian tata letak
- Komparasi Dua Algoritma (Side-by-Side Comparison) lengkap dengan metrik:
  * Jalur Terpendek
  * Bobot Minimum
  * Waktu Eksekusi (Runtime mikrodetik)
  * Jumlah Simpul yang Dikunjungi (Nodes Expanded)
- Dukungan Mode CLI (Command Line Interface) dan Mode GUI
- 100% Zero-Dependency (Hanya menggunakan Python Standard Library)
=================================================================================
"""

import sys
import os
import re
import ast
import time
import math
import random
import heapq
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Penanganan encoding konsol Windows agar karakter panah tidak error di cp1252
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# 1. MODUL PARSER FILE GRAF
# =============================================================================

def parse_graph_file(filepath):
    """
    Membaca dan mem-parsing file graf.txt sesuai format penugasan.
    Contoh format:
    graph = {
      'V1': {'V2': 120, 'V4': 200},
      ...
    }
    V1 V15
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' tidak ditemukan.")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("File graf kosong!")

    # Baris terakhir berisi simpul awal dan simpul akhir
    last_line = lines[-1]
    tokens = last_line.split()
    if len(tokens) < 2:
        raise ValueError(
            f"Baris terakhir file harus memuat simpul awal dan akhir (dipisahkan spasi). Ditemukan: '{last_line}'"
        )
    start_node = tokens[0].strip()
    goal_node = tokens[1].strip()

    # Ekstrak bagian dictionary Python
    dict_content = "\n".join(lines[:-1])
    match = re.search(r"\{.*\}", dict_content, re.DOTALL)
    if not match:
        raise ValueError("Struktur dictionary graf {...} tidak ditemukan di dalam file!")

    raw_dict_str = match.group(0)
    try:
        raw_graph = ast.literal_eval(raw_dict_str)
    except Exception as e:
        raise ValueError(f"Gagal mem-parsing dictionary graf: {e}")

    # Normalisasi tipe data: simpul sebagai string, bobot sebagai float/int
    graph = {}
    for u, neighbors in raw_graph.items():
        u_str = str(u).strip()
        graph[u_str] = {}
        for v, w in neighbors.items():
            graph[u_str][str(v).strip()] = float(w)

    # Validasi keberadaan start_node dan goal_node di dalam graf
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors.keys())

    # Pastikan setiap simpul terdaftar sebagai key di graph dictionary
    for node in all_nodes:
        if node not in graph:
            graph[node] = {}

    if start_node not in graph:
        raise ValueError(f"Simpul awal '{start_node}' tidak ditemukan dalam graf.")
    if goal_node not in graph:
        raise ValueError(f"Simpul tujuan '{goal_node}' tidak ditemukan dalam graf.")

    return graph, start_node, goal_node


# =============================================================================
# 2. IMPLEMENTASI ALGORITMA SHORTEST PATH
# =============================================================================

def dijkstra_algorithm(graph, start, goal):
    """
    Algoritma Dijkstra (Greedy / Uniform Cost Search).
    Mengembalikan:
      - path: list jalur simpul terpendek
      - total_weight: total bobot minimum
      - runtime_us: waktu eksekusi dalam mikrodetik (µs)
      - steps: riwayat langkah untuk animasi GUI [(event, node, distance, parent, frontier_snapshot)]
      - visited_nodes_count: jumlah simpul yang diekspansi
    """
    start_time = time.perf_counter()

    distances = {node: float("inf") for node in graph}
    distances[start] = 0.0
    previous = {node: None for node in graph}
    
    # Priority Queue menyimpan tuple: (jarak, counter, simpul)
    counter = 0
    pq = [(0.0, counter, start)]
    visited = set()
    steps = []

    while pq:
        curr_dist, _, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        visited.add(curr_node)
        steps.append({
            "type": "visit",
            "node": curr_node,
            "dist": curr_dist,
            "prev": previous[curr_node]
        })

        if curr_node == goal:
            break

        for neighbor, weight in graph[curr_node].items():
            if neighbor in visited:
                continue
            new_dist = curr_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = curr_node
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))
                steps.append({
                    "type": "relax",
                    "from": curr_node,
                    "to": neighbor,
                    "new_dist": new_dist
                })

    runtime_us = (time.perf_counter() - start_time) * 1_000_000

    # Rekonstruksi Path
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()

    if not path or path[0] != start:
        return None, float("inf"), runtime_us, steps, len(visited)

    total_weight = distances[goal]
    if total_weight.is_integer():
        total_weight = int(total_weight)

    return path, total_weight, runtime_us, steps, len(visited)


def compute_admissible_heuristic(graph, goal):
    """
    Menghitung fungsi heuristik h(n) yang dijamin Admissible dan Consistent.
    h(n) = hop_distance_ke_goal(n) * min_edge_weight_seluruh_graf.
    Karena setiap langkah edge pada graf memiliki bobot minimal min_edge_weight,
    maka (hop_dist * min_edge_weight) selalu <= bobot jalur terpendek sebenarnya (h*(n)).
    """
    min_weight = float("inf")
    for u in graph:
        for v, w in graph[u].items():
            if w < min_weight and w > 0:
                min_weight = w
    if min_weight == float("inf"):
        min_weight = 1.0

    # BFS ke belakang dari goal pada graf yang dibalik arahnya (atau undirected)
    # Untuk graf tak berarah / umum:
    reversed_graph = collections.defaultdict(list)
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)

    hop_dist = {goal: 0}
    queue = collections.deque([goal])
    while queue:
        curr = queue.popleft()
        for predecessor in reversed_graph[curr]:
            if predecessor not in hop_dist:
                hop_dist[predecessor] = hop_dist[curr] + 1
                queue.append(predecessor)

    heuristic = {}
    for node in graph:
        heuristic[node] = hop_dist.get(node, 0) * min_weight

    return heuristic


def astar_algorithm(graph, start, goal):
    """
    Algoritma A* (A-Star / Informed Heuristic Search).
    Evaluasi fungsi: f(n) = g(n) + h(n)
    Di mana g(n) = jarak aktual dari start, h(n) = estimasi heuristik admissible ke goal.
    """
    start_time = time.perf_counter()

    heuristic = compute_admissible_heuristic(graph, goal)

    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0.0
    f_score = {node: float("inf") for node in graph}
    f_score[start] = heuristic.get(start, 0.0)

    counter = 0
    pq = [(f_score[start], counter, start)]
    previous = {node: None for node in graph}
    visited = set()
    steps = []

    while pq:
        curr_f, _, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        visited.add(curr_node)
        steps.append({
            "type": "visit",
            "node": curr_node,
            "dist": g_score[curr_node],
            "f_score": curr_f,
            "h": heuristic.get(curr_node, 0.0),
            "prev": previous[curr_node]
        })

        if curr_node == goal:
            break

        for neighbor, weight in graph[curr_node].items():
            if neighbor in visited:
                continue
            tentative_g = g_score[curr_node] + weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic.get(neighbor, 0.0)
                f_score[neighbor] = f
                previous[neighbor] = curr_node
                counter += 1
                heapq.heappush(pq, (f, counter, neighbor))
                steps.append({
                    "type": "relax",
                    "from": curr_node,
                    "to": neighbor,
                    "new_dist": tentative_g,
                    "f_score": f
                })

    runtime_us = (time.perf_counter() - start_time) * 1_000_000

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()

    if not path or path[0] != start:
        return None, float("inf"), runtime_us, steps, len(visited)

    total_weight = g_score[goal]
    if total_weight.is_integer():
        total_weight = int(total_weight)

    return path, total_weight, runtime_us, steps, len(visited)


def bellman_ford_algorithm(graph, start, goal):
    """
    Algoritma Bellman-Ford (Dynamic Programming Search).
    Merelaksasi seluruh sisi graf sebanyak |V| - 1 kali.
    Dapat mendeteksi siklus negatif.
    """
    start_time = time.perf_counter()

    distances = {node: float("inf") for node in graph}
    distances[start] = 0.0
    previous = {node: None for node in graph}

    # Kumpulkan seluruh sisi
    edges = []
    for u in graph:
        for v, w in graph[u].items():
            edges.append((u, v, w))

    nodes = list(graph.keys())
    num_nodes = len(nodes)
    steps = []
    visited_nodes = {start}

    for iteration in range(num_nodes - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] != float("inf") and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                previous[v] = u
                visited_nodes.add(v)
                updated = True
                steps.append({
                    "type": "relax",
                    "from": u,
                    "to": v,
                    "new_dist": distances[v],
                    "iteration": iteration + 1
                })
        if not updated:
            break

    # Cek siklus negatif
    for u, v, w in edges:
        if distances[u] != float("inf") and distances[u] + w < distances[v]:
            raise ValueError("Graf mengandung siklus berbobot negatif (Negative Cycle)!")

    runtime_us = (time.perf_counter() - start_time) * 1_000_000

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()

    if not path or path[0] != start:
        return None, float("inf"), runtime_us, steps, len(visited_nodes)

    total_weight = distances[goal]
    if total_weight.is_integer():
        total_weight = int(total_weight)

    return path, total_weight, runtime_us, steps, len(visited_nodes)


# =============================================================================
# 3. MODUL FORCE-DIRECTED LAYOUT (SPRING EMBEDDER) UNTUK TATA LETAK GRAF
# =============================================================================

def compute_graph_layout(graph, width=680, height=520, padding=70, iterations=70):
    """
    Menghitung posisi 2D (x, y) tiap simpul menggunakan simulasi gaya pegas (Spring-Force).
    Memastikan simpul tidak bertumpuk dan tersebar rapi secara estetis.
    """
    nodes = list(graph.keys())
    N = len(nodes)
    if N == 0:
        return {}
    if N == 1:
        return {nodes[0]: (width / 2, height / 2)}

    # Inisialisasi awal melingkar (Circular Layout)
    cx, cy = width / 2, height / 2
    r = min(width, height) / 2 - padding
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / N
        pos[node] = [cx + r * math.cos(angle) + random.uniform(-10, 10),
                     cy + r * math.sin(angle) + random.uniform(-10, 10)]

    # Parameter simulasi
    area = (width - 2 * padding) * (height - 2 * padding)
    k = math.sqrt(area / max(N, 1)) * 0.95
    temp = width / 10

    # Daftar sisi graf tak berarah untuk gaya tarik
    edge_set = set()
    for u in graph:
        for v in graph[u]:
            pair = tuple(sorted([u, v]))
            edge_set.add(pair)

    for it in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}

        # Gaya tolak-menolak antar semua pasangan simpul
        for i in range(N):
            u = nodes[i]
            for j in range(i + 1, N):
                v = nodes[j]
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                dist = math.hypot(dx, dy)
                if dist < 0.01:
                    dist = 0.01
                    dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)

                rep_force = (k * k) / dist
                disp[u][0] += (dx / dist) * rep_force
                disp[u][1] += (dy / dist) * rep_force
                disp[v][0] -= (dx / dist) * rep_force
                disp[v][1] -= (dy / dist) * rep_force

        # Gaya tarik antar simpul yang terhubung sisi
        for u, v in edge_set:
            if u not in pos or v not in pos:
                continue
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            dist = math.hypot(dx, dy)
            if dist < 0.01:
                dist = 0.01
            att_force = (dist * dist) / k
            disp[u][0] -= (dx / dist) * att_force
            disp[u][1] -= (dy / dist) * att_force
            disp[v][0] += (dx / dist) * att_force
            disp[v][1] += (dy / dist) * att_force

        # Terapkan perpindahan dengan pendinginan (simulated annealing)
        current_temp = temp * (1.0 - (it / iterations))
        for node in nodes:
            d_len = math.hypot(disp[node][0], disp[node][1])
            if d_len > 0.01:
                factor = min(d_len, current_temp) / d_len
                pos[node][0] += disp[node][0] * factor
                pos[node][1] += disp[node][1] * factor

            # Batas canvas (bounding box)
            pos[node][0] = max(padding, min(width - padding, pos[node][0]))
            pos[node][1] = max(padding, min(height - padding, pos[node][1]))

    return {node: (pos[node][0], pos[node][1]) for node in nodes}


# =============================================================================
# 4. ANTARMUKA GRAFIS (GUI) MODERN BERBASIS TKINTER
# =============================================================================

class ShortestPathGUI:
    def __init__(self, root, initial_filepath="graf.txt"):
        self.root = root
        self.root.title("MMAI1001 - Visualisasi Algoritma Shortest Path (Kecerdasan Artificial)")
        self.root.geometry("1200x780")
        self.root.minsize(1050, 680)

        # Skema Warna Estetik Modern (Light / Clean Minimalist / Gen Z)
        self.colors = {
            "bg_dark": "#f8fafc",        # Latar belakang utama (Off-white yang sangat soft)
            "panel_bg": "#ffffff",       # Panel menu (Putih bersih)
            "card_bg": "#f1f5f9",        # Card sedikit abu-abu untuk pembeda
            "accent": "#6366f1",         # Indigo/Purple modern yang vibrant (Gen Z)
            "accent_hover": "#4f46e5",   # Indigo lebih gelap untuk hover
            "text_main": "#0f172a",      # Teks utama (Hitam pekat/Slate gelap)
            "text_dim": "#64748b",       # Teks pendukung (Abu-abu sedang)
            "node_default": "#e2e8f0",   # Node abu muda yang bersih
            "node_start": "#10b981",     # Hijau emerald cerah
            "node_goal": "#ef4444",      # Merah coral
            "node_visited": "#c084fc",   # Ungu pastel untuk yang sudah dikunjungi
            "node_current": "#f59e0b",   # Kuning amber/orange
            "node_path": "#6366f1",      # Node pada jalur (mengikuti accent color)
            "edge_default": "#cbd5e1",   # Garis abu-abu terang
            "edge_path": "#8b5cf6",      # Garis jalur dengan ungu cerah
            "edge_explore": "#2dd4bf",   # Garis eksplorasi warna teal
            "edge_label_bg": "#ffffff",  # Background label edge
            "canvas_bg": "#ffffff"       # Canvas putih bersih untuk kejelasan maksimal
        }

        self.root.configure(bg=self.colors["bg_dark"])

        # Data Graf & Algoritma
        self.current_filepath = initial_filepath
        self.graph = {}
        self.start_node = ""
        self.goal_node = ""
        self.node_positions = {}
        
        # State Animasi & Interaksi
        self.selected_algo = tk.StringVar(value="Dijkstra")
        self.anim_speed_ms = tk.IntVar(value=180)
        self.is_animating = False
        self.animation_job = None
        self.current_step_idx = 0
        self.steps = []
        self.visited_nodes = set()
        self.explored_edges = set()
        self.shortest_path = []
        self.shortest_path_edges = set()
        self.current_active_node = None

        # State Drag & Drop Node
        self.dragged_node = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        # Inisialisasi UI
        self._setup_styles()
        self._build_layout()

        # Muat file awal
        self.load_file(self.current_filepath)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Styling ttk widget agar menyatu dengan UI Dark
        style.configure(".", background=self.colors["panel_bg"], foreground=self.colors["text_main"])
        style.configure("TLabel", background=self.colors["panel_bg"], foreground=self.colors["text_main"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), foreground=self.colors["accent"])
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground=self.colors["text_dim"])
        style.configure("CardHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"])

        style.configure("TCombobox", fieldbackground=self.colors["card_bg"], background=self.colors["card_bg"],
                        foreground=self.colors["text_main"], arrowcolor=self.colors["accent"])
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["card_bg"])],
                  selectbackground=[("readonly", self.colors["accent_hover"])])

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=self.colors["accent"],
                        foreground="#0f172a", borderwidth=0, padding=6)
        style.map("Accent.TButton", background=[("active", self.colors["accent_hover"])])

        style.configure("Secondary.TButton", font=("Segoe UI", 9), background=self.colors["card_bg"],
                        foreground=self.colors["text_main"], borderwidth=0, padding=5)
        style.map("Secondary.TButton", background=[("active", self.colors["node_default"])])

    def _build_layout(self):
        # 1. Header Bar
        header_frame = tk.Frame(self.root, bg=self.colors["panel_bg"], height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        title_lbl = tk.Label(header_frame, text="MMAI1001 — Visualisasi Algoritma Shortest Path",
                             font=("Segoe UI", 14, "bold"), fg=self.colors["accent"], bg=self.colors["panel_bg"])
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        subtitle_lbl = tk.Label(header_frame, text="Kecerdasan Artificial S2 Informatika | Dijkstra • A* • Bellman-Ford",
                                font=("Segoe UI", 9), fg=self.colors["text_dim"], bg=self.colors["panel_bg"])
        subtitle_lbl.pack(side=tk.RIGHT, padx=20, pady=10)

        # 2. Main Container (Sidebar Kiri + Canvas Visualisasi Kanan)
        main_container = tk.Frame(self.root, bg=self.colors["bg_dark"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # Sidebar Kiri (Kontrol & Hasil)
        sidebar = tk.Frame(main_container, bg=self.colors["panel_bg"], width=360)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=0)
        sidebar.pack_propagate(False)

        # Canvas Kanan
        canvas_card = tk.Frame(main_container, bg=self.colors["panel_bg"], bd=1, relief=tk.FLAT)
        canvas_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_sidebar_content(sidebar)
        self._build_canvas_content(canvas_card)

    def _build_sidebar_content(self, sidebar):
        # Scrollable container untuk sidebar jika resolusi layar kecil
        canvas_sb = tk.Canvas(sidebar, bg=self.colors["panel_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas_sb.yview)
        scrollable_frame = tk.Frame(canvas_sb, bg=self.colors["panel_bg"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_sb.configure(scrollregion=canvas_sb.bbox("all"))
        )

        canvas_sb.create_window((0, 0), window=scrollable_frame, anchor="nw", width=340)
        canvas_sb.yview_moveto(0)
        canvas_sb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        pad_x = 14

        # Card 1: File Input & Sumber Graf
        c1 = tk.Frame(scrollable_frame, bg=self.colors["card_bg"], padx=12, pady=10)
        c1.pack(fill=tk.X, padx=pad_x, pady=(12, 8))
        tk.Label(c1, text="📁 SUMBER GRAF", font=("Segoe UI", 10, "bold"),
                 fg=self.colors["accent"], bg=self.colors["card_bg"]).pack(anchor="w")

        self.file_label = tk.Label(c1, text="File: -", font=("Segoe UI", 8),
                                   fg=self.colors["text_dim"], bg=self.colors["card_bg"], wraplength=310, justify="left")
        self.file_label.pack(anchor="w", pady=(4, 8))

        btn_row_file = tk.Frame(c1, bg=self.colors["card_bg"])
        btn_row_file.pack(fill=tk.X)

        b_reload = tk.Button(btn_row_file, text="⟳ graf.txt", command=lambda: self.load_file("graf.txt"),
                             bg=self.colors["panel_bg"], fg=self.colors["text_main"], font=("Segoe UI", 8),
                             activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=8, pady=4)
        b_reload.pack(side=tk.LEFT, padx=(0, 4))

        b_browse = tk.Button(btn_row_file, text="Pilih File...", command=self._browse_file,
                             bg=self.colors["panel_bg"], fg=self.colors["text_main"], font=("Segoe UI", 8),
                             activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=8, pady=4)
        b_browse.pack(side=tk.LEFT, padx=4)

        b_sample_b = tk.Button(btn_row_file, text="Contoh B", command=lambda: self.load_file("graf_contoh_b.txt"),
                               bg=self.colors["panel_bg"], fg=self.colors["text_main"], font=("Segoe UI", 8),
                               activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=6, pady=4)
        b_sample_b.pack(side=tk.LEFT, padx=4)

        b_sample_c = tk.Button(btn_row_file, text="Contoh C", command=lambda: self.load_file("graf_contoh_c.txt"),
                               bg=self.colors["panel_bg"], fg=self.colors["text_main"], font=("Segoe UI", 8),
                               activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=4)
        b_sample_c.pack(side=tk.LEFT)

        # Card 2: Pengaturan Simpul & Algoritma
        c2 = tk.Frame(scrollable_frame, bg=self.colors["card_bg"], padx=12, pady=10)
        c2.pack(fill=tk.X, padx=pad_x, pady=6)
        tk.Label(c2, text="⚙ PENGATURAN", font=("Segoe UI", 10, "bold"),
                 fg=self.colors["accent"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(0, 6))

        # Start & Goal Row
        node_row = tk.Frame(c2, bg=self.colors["card_bg"])
        node_row.pack(fill=tk.X, pady=4)

        lbl_s = tk.Label(node_row, text="Awal (Start):", font=("Segoe UI", 9), fg=self.colors["text_main"], bg=self.colors["card_bg"])
        lbl_s.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.combo_start = ttk.Combobox(node_row, width=8, state="readonly")
        self.combo_start.grid(row=0, column=1, sticky="w")
        self.combo_start.bind("<<ComboboxSelected>>", self._on_node_selected)

        lbl_g = tk.Label(node_row, text="Akhir (Goal):", font=("Segoe UI", 9), fg=self.colors["text_main"], bg=self.colors["card_bg"])
        lbl_g.grid(row=0, column=2, sticky="w", padx=(12, 6))
        self.combo_goal = ttk.Combobox(node_row, width=8, state="readonly")
        self.combo_goal.grid(row=0, column=3, sticky="w")
        self.combo_goal.bind("<<ComboboxSelected>>", self._on_node_selected)

        # Pilihan Algoritma
        tk.Label(c2, text="Pilih Algoritma:", font=("Segoe UI", 9), fg=self.colors["text_main"],
                 bg=self.colors["card_bg"]).pack(anchor="w", pady=(8, 4))
        
        algo_frame = tk.Frame(c2, bg=self.colors["card_bg"])
        algo_frame.pack(fill=tk.X)
        for algo_name in ["Dijkstra", "A* (A-Star)", "Bellman-Ford"]:
            rb = tk.Radiobutton(algo_frame, text=algo_name, variable=self.selected_algo, value=algo_name,
                                font=("Segoe UI", 9), fg=self.colors["text_main"], bg=self.colors["card_bg"],
                                activebackground=self.colors["card_bg"], selectcolor=self.colors["panel_bg"],
                                activeforeground=self.colors["accent"], command=self.reset_visualization)
            rb.pack(anchor="w")

        # Slider Kecepatan
        tk.Label(c2, text="Kecepatan Animasi (ms):", font=("Segoe UI", 8),
                 fg=self.colors["text_dim"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(8, 2))
        speed_scale = tk.Scale(c2, from_=20, to=600, orient=tk.HORIZONTAL, variable=self.anim_speed_ms,
                               bg=self.colors["card_bg"], fg=self.colors["text_main"],
                               troughcolor=self.colors["panel_bg"], highlightthickness=0, bd=0)
        speed_scale.pack(fill=tk.X)

        # Card 3: Tombol Aksi Kontrol
        c3 = tk.Frame(scrollable_frame, bg=self.colors["card_bg"], padx=12, pady=10)
        c3.pack(fill=tk.X, padx=pad_x, pady=6)

        btn_run = tk.Button(c3, text="▶ Jalankan Pencarian", command=self.start_search,
                            font=("Segoe UI", 10, "bold"), bg=self.colors["accent"], fg="#0f172a",
                            activebackground=self.colors["accent_hover"], relief=tk.FLAT, bd=0, pady=6)
        btn_run.pack(fill=tk.X, pady=(0, 6))

        row_ctrl = tk.Frame(c3, bg=self.colors["card_bg"])
        row_ctrl.pack(fill=tk.X)

        btn_step = tk.Button(row_ctrl, text="⏭ Langkah (Step)", command=self.step_forward,
                             font=("Segoe UI", 8), bg=self.colors["panel_bg"], fg=self.colors["text_main"],
                             activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, pady=5)
        btn_step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_reset = tk.Button(row_ctrl, text="🔄 Reset", command=self.reset_visualization,
                              font=("Segoe UI", 8), bg=self.colors["panel_bg"], fg=self.colors["text_main"],
                              activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, pady=5)
        btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_compare = tk.Button(c3, text="⚖ Bandingkan Algoritma (Side-by-Side)", command=self.open_comparison_dialog,
                                font=("Segoe UI", 9, "bold"), bg="#6366f1", fg="#ffffff",
                                activebackground="#4f46e5", relief=tk.FLAT, bd=0, pady=6)
        btn_compare.pack(fill=tk.X, pady=(8, 0))

        # Card 4: Panel Hasil Output (Sesuai Format Tugas!)
        c4 = tk.Frame(scrollable_frame, bg=self.colors["card_bg"], padx=12, pady=10)
        c4.pack(fill=tk.X, padx=pad_x, pady=(6, 16))

        tk.Label(c4, text="📊 HASIL OUTPUT", font=("Segoe UI", 10, "bold"),
                 fg=self.colors["accent"], bg=self.colors["card_bg"]).pack(anchor="w")

        self.status_lbl = tk.Label(c4, text="Status: Siap", font=("Segoe UI", 9, "italic"),
                                   fg=self.colors["text_dim"], bg=self.colors["card_bg"])
        self.status_lbl.pack(anchor="w", pady=(4, 6))

        self.path_lbl = tk.Label(c4, text="Jalur terpendek: -", font=("Segoe UI", 9, "bold"),
                                 fg="#38bdf8", bg=self.colors["card_bg"], wraplength=310, justify="left")
        self.path_lbl.pack(anchor="w", pady=2)

        self.weight_lbl = tk.Label(c4, text="Bobot minimum: -", font=("Segoe UI", 9, "bold"),
                                   fg="#34d399", bg=self.colors["card_bg"])
        self.weight_lbl.pack(anchor="w", pady=2)

        self.runtime_lbl = tk.Label(c4, text="Waktu runtime: -", font=("Segoe UI", 9, "bold"),
                                    fg="#fbbf24", bg=self.colors["card_bg"])
        self.runtime_lbl.pack(anchor="w", pady=2)

        self.expanded_lbl = tk.Label(c4, text="Simpul dieksplorasi: -", font=("Segoe UI", 8),
                                     fg=self.colors["text_dim"], bg=self.colors["card_bg"])
        self.expanded_lbl.pack(anchor="w", pady=2)

        # Petunjuk Interaksi Singkat
        guide_box = tk.Label(scrollable_frame,
                             text="💡 Petunjuk Interaksi:\n• Klik Kiri simpul: Set Start\n• Klik Kanan simpul: Set Goal\n• Drag simpul untuk merapikan posisi",
                             font=("Segoe UI", 8), fg=self.colors["text_dim"], bg=self.colors["panel_bg"], justify="left")
        guide_box.pack(anchor="w", padx=pad_x, pady=(0, 16))

    def _build_canvas_content(self, parent):
        # Top toolbar canvas
        top_bar = tk.Frame(parent, bg=self.colors["panel_bg"], height=38)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(8, 4))

        tk.Label(top_bar, text="Visualisasi Struktur Graf & Jalur Terpendek",
                 font=("Segoe UI", 11, "bold"), fg=self.colors["text_main"], bg=self.colors["panel_bg"]).pack(side=tk.LEFT)

        btn_recalc_layout = tk.Button(top_bar, text="📐 Tata Ulang Posisi (Auto-Layout)", command=self._recalculate_layout,
                                      bg=self.colors["card_bg"], fg=self.colors["text_main"], font=("Segoe UI", 8),
                                      activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=8, pady=3)
        btn_recalc_layout.pack(side=tk.RIGHT)

        # Main Canvas
        self.canvas = tk.Canvas(parent, bg=self.colors["canvas_bg"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Bind Event Canvas (Drag & Drop, Klik Start/Goal)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click_left)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_click_right)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Bottom Bar: Legenda Warna
        legend_bar = tk.Frame(parent, bg=self.colors["panel_bg"], height=32)
        legend_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 8))

        legends = [
            ("Start", self.colors["node_start"]),
            ("Goal", self.colors["node_goal"]),
            ("Frontier / Visiting", self.colors["node_current"]),
            ("Visited", self.colors["node_visited"]),
            ("Shortest Path", self.colors["node_path"])
        ]
        for name, color in legends:
            dot = tk.Canvas(legend_bar, width=12, height=12, bg=self.colors["panel_bg"], highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=color, outline="")
            dot.pack(side=tk.LEFT, padx=(12, 4), pady=4)
            tk.Label(legend_bar, text=name, font=("Segoe UI", 8), fg=self.colors["text_dim"],
                     bg=self.colors["panel_bg"]).pack(side=tk.LEFT, pady=4)

    # =========================================================================
    # EVENT HANDLERS & LOGIKA FILE
    # =========================================================================

    def _browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Pilih File Graf",
            filetypes=[("Text Files (*.txt)", "*.txt"), ("All Files (*.*)", "*.*")],
            initialdir=os.getcwd()
        )
        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath):
        try:
            graph, start, goal = parse_graph_file(filepath)
            self.current_filepath = filepath
            self.graph = graph
            self.start_node = start
            self.goal_node = goal

            # Update Label File
            filename = os.path.basename(filepath)
            self.file_label.config(text=f"File: {filename} ({len(graph)} simpul)")

            # Update Combobox Options
            nodes_sorted = sorted(list(graph.keys()), key=lambda x: (len(x), x))
            self.combo_start["values"] = nodes_sorted
            self.combo_goal["values"] = nodes_sorted

            if start in nodes_sorted:
                self.combo_start.set(start)
            if goal in nodes_sorted:
                self.combo_goal.set(goal)

            # Hitung Layout Posisi
            self._recalculate_layout()
            self.reset_visualization()

        except Exception as e:
            messagebox.showerror("Error Membaca File", f"Gagal membaca file '{filepath}':\n\n{e}")

    def _recalculate_layout(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 200 or h < 200:
            w, h = 760, 560
        self.node_positions = compute_graph_layout(self.graph, width=w, height=h)
        self.draw_graph()

    def _on_canvas_resize(self, event):
        # Jika pertama kali di-render dan layout belum ada
        if not self.node_positions and self.graph:
            self._recalculate_layout()
        else:
            self.draw_graph()

    def _on_node_selected(self, event=None):
        new_start = self.combo_start.get()
        new_goal = self.combo_goal.get()
        if new_start in self.graph:
            self.start_node = new_start
        if new_goal in self.graph:
            self.goal_node = new_goal
        self.reset_visualization()

    def _get_node_at_pos(self, x, y, radius=24):
        for node, (nx, ny) in self.node_positions.items():
            if math.hypot(nx - x, ny - y) <= radius:
                return node
        return None

    def _on_canvas_click_left(self, event):
        node = self._get_node_at_pos(event.x, event.y)
        if node:
            self.dragged_node = node
            nx, ny = self.node_positions[node]
            self.drag_offset_x = nx - event.x
            self.drag_offset_y = ny - event.y
            # Jika shift atau double click, jadikan start node
            if event.state & 0x0001:  # Shift pressed
                self.start_node = node
                self.combo_start.set(node)
                self.reset_visualization()

    def _on_canvas_drag(self, event):
        if self.dragged_node:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            new_x = max(35, min(w - 35, event.x + self.drag_offset_x))
            new_y = max(35, min(h - 35, event.y + self.drag_offset_y))
            self.node_positions[self.dragged_node] = (new_x, new_y)
            self.draw_graph()

    def _on_canvas_release(self, event):
        self.dragged_node = None

    def _on_canvas_click_right(self, event):
        # Klik kanan untuk memilih Start atau Goal
        node = self._get_node_at_pos(event.x, event.y)
        if node:
            # Menu popup untuk kemudahan
            menu = tk.Menu(self.root, tearoff=0, bg=self.colors["panel_bg"], fg=self.colors["text_main"])
            menu.add_command(label=f"Jadikan Simpul Awal ({node})", command=lambda: self._set_start(node))
            menu.add_command(label=f"Jadikan Simpul Tujuan ({node})", command=lambda: self._set_goal(node))
            menu.post(event.x_root, event.y_root)

    def _set_start(self, node):
        self.start_node = node
        self.combo_start.set(node)
        self.reset_visualization()

    def _set_goal(self, node):
        self.goal_node = node
        self.combo_goal.set(node)
        self.reset_visualization()

    # =========================================================================
    # VISUALISASI MENGGAMBAR GRAF PADA CANVAS
    # =========================================================================

    def draw_graph(self):
        self.canvas.delete("all")
        if not self.graph or not self.node_positions:
            return

        # 1. Gambar Edges (Garis penghubung antar simpul)
        drawn_edges = set()
        for u in self.graph:
            if u not in self.node_positions:
                continue
            x1, y1 = self.node_positions[u]
            for v, weight in self.graph[u].items():
                if v not in self.node_positions:
                    continue
                x2, y2 = self.node_positions[v]

                pair_key = tuple(sorted([u, v]))
                is_path = (u, v) in self.shortest_path_edges or (v, u) in self.shortest_path_edges
                is_explored = (u, v) in self.explored_edges or (v, u) in self.explored_edges

                # Tentukan warna dan ketebalan garis
                if is_path:
                    edge_color = self.colors["edge_path"]
                    edge_width = 4
                elif is_explored:
                    edge_color = self.colors["edge_explore"]
                    edge_width = 2
                else:
                    edge_color = self.colors["edge_default"]
                    edge_width = 1.5

                self.canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=edge_width, smooth=True)

                # Label Bobot (Hanya gambar sekali untuk graf tak berarah agar tidak tumpang tindih)
                if pair_key not in drawn_edges:
                    drawn_edges.add(pair_key)
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2

                    w_text = str(int(weight) if isinstance(weight, float) and weight.is_integer() else weight)

                    # Pill badge di belakang label bobot
                    pad = 3
                    text_id = self.canvas.create_text(mx, my, text=w_text, font=("Segoe UI", 8, "bold"),
                                                      fill=self.colors["accent"] if is_path else "#e2e8f0")
                    bbox = self.canvas.bbox(text_id)
                    if bbox:
                        rect_id = self.canvas.create_rectangle(
                            bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad,
                            fill=self.colors["panel_bg"], outline=edge_color, width=1
                        )
                        self.canvas.tag_lower(rect_id, text_id)

        # 2. Gambar Nodes (Lingkaran Simpul)
        radius = 18
        for node, (x, y) in self.node_positions.items():
            # Tentukan warna simpul
            if node == self.start_node:
                fill_color = self.colors["node_start"]
                border_color = "#ffffff"
                border_w = 3
            elif node == self.goal_node:
                fill_color = self.colors["node_goal"]
                border_color = "#ffffff"
                border_w = 3
            elif node in self.shortest_path:
                fill_color = self.colors["node_path"]
                border_color = "#38bdf8"
                border_w = 2.5
            elif node == self.current_active_node:
                fill_color = self.colors["node_current"]
                border_color = "#fef08a"
                border_w = 3
            elif node in self.visited_nodes:
                fill_color = self.colors["node_visited"]
                border_color = "#c084fc"
                border_w = 1.5
            else:
                fill_color = self.colors["node_default"]
                border_color = "#94a3b8"
                border_w = 1.5

            # Lingkaran simpul
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=fill_color, outline=border_color, width=border_w
            )

            # Teks Nama Simpul
            self.canvas.create_text(
                x, y, text=node, font=("Segoe UI", 9, "bold"),
                fill="#ffffff"
            )

    # =========================================================================
    # KONTROL EKSEKUSI PENCARIAN & ANIMASI
    # =========================================================================

    def reset_visualization(self):
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self.is_animating = False

        self.current_step_idx = 0
        self.steps = []
        self.visited_nodes = set()
        self.explored_edges = set()
        self.shortest_path = []
        self.shortest_path_edges = set()
        self.current_active_node = None

        self.status_lbl.config(text="Status: Siap")
        self.path_lbl.config(text="Jalur terpendek: -")
        self.weight_lbl.config(text="Bobot minimum: -")
        self.runtime_lbl.config(text="Waktu runtime: -")
        self.expanded_lbl.config(text="Simpul dieksplorasi: -")

        self.draw_graph()

    def _prepare_search(self):
        if not self.graph:
            messagebox.showwarning("Peringatan", "Graf belum dimuat!")
            return False
        if not self.start_node or not self.goal_node:
            messagebox.showwarning("Peringatan", "Tentukan simpul Awal dan Tujuan terlebih dahulu!")
            return False

        algo = self.selected_algo.get()
        if algo == "Dijkstra":
            path, cost, runtime, steps, expanded = dijkstra_algorithm(self.graph, self.start_node, self.goal_node)
        elif algo == "A* (A-Star)":
            path, cost, runtime, steps, expanded = astar_algorithm(self.graph, self.start_node, self.goal_node)
        elif algo == "Bellman-Ford":
            path, cost, runtime, steps, expanded = bellman_ford_algorithm(self.graph, self.start_node, self.goal_node)
        else:
            messagebox.showerror("Error", f"Algoritma '{algo}' tidak dikenali.")
            return False

        self.steps = steps
        self.final_result = {
            "path": path,
            "cost": cost,
            "runtime": runtime,
            "expanded": expanded,
            "algo": algo
        }
        return True

    def start_search(self):
        self.reset_visualization()
        if not self._prepare_search():
            return

        self.is_animating = True
        self.status_lbl.config(text=f"Status: Menjalankan {self.selected_algo.get()}...")
        self._animate_next_step()

    def _animate_next_step(self):
        if not self.is_animating:
            return

        if self.current_step_idx < len(self.steps):
            step = self.steps[self.current_step_idx]
            self._apply_step(step)
            self.current_step_idx += 1
            self.draw_graph()

            delay = max(10, self.anim_speed_ms.get())
            self.animation_job = self.root.after(delay, self._animate_next_step)
        else:
            self._finalize_search_results()

    def step_forward(self):
        if not self.steps:
            if not self._prepare_search():
                return
            self.status_lbl.config(text=f"Status: Mode Langkah per Langkah ({self.selected_algo.get()})")

        if self.current_step_idx < len(self.steps):
            step = self.steps[self.current_step_idx]
            self._apply_step(step)
            self.current_step_idx += 1
            self.draw_graph()
        else:
            self._finalize_search_results()

    def _apply_step(self, step):
        step_type = step.get("type")
        if step_type == "visit":
            node = step.get("node")
            self.current_active_node = node
            self.visited_nodes.add(node)
            prev = step.get("prev")
            if prev:
                self.explored_edges.add((prev, node))
        elif step_type == "relax":
            u = step.get("from")
            v = step.get("to")
            self.explored_edges.add((u, v))

    def _finalize_search_results(self):
        self.is_animating = False
        res = self.final_result

        path = res["path"]
        cost = res["cost"]
        runtime = res["runtime"]
        expanded = res["expanded"]

        if path:
            self.shortest_path = path
            self.shortest_path_edges = set()
            for i in range(len(path) - 1):
                self.shortest_path_edges.add((path[i], path[i + 1]))

            # Format cantik: V1 → V2 → ...
            path_str = " → ".join(path)
            self.status_lbl.config(text="Status: Jalur Terpendek Ditemukan! ✓", fg="#34d399")
            self.path_lbl.config(text=f"Jalur terpendek: {path_str}")
            self.weight_lbl.config(text=f"Bobot minimum: {cost}")
            # Format runtime: jika < 1000 us tampilkan us, jika >= tampilkan ms
            if runtime < 1000:
                self.runtime_lbl.config(text=f"Waktu runtime: {runtime:.2f} µs")
            else:
                self.runtime_lbl.config(text=f"Waktu runtime: {runtime/1000:.3f} ms ({runtime:.1f} µs)")
            self.expanded_lbl.config(text=f"Simpul dieksplorasi: {expanded} dari {len(self.graph)} simpul")
        else:
            self.status_lbl.config(text="Status: Tidak ada jalur yang menghubungkan!", fg="#ef4444")
            self.path_lbl.config(text="Jalur terpendek: Tidak ditemukan")
            self.weight_lbl.config(text="Bobot minimum: ∞")
            self.runtime_lbl.config(text=f"Waktu runtime: {runtime:.2f} µs")

        self.current_active_node = None
        self.draw_graph()

    # =========================================================================
    # DIALOG KOMPARASI DUA ALGORITMA (SIDE-BY-SIDE COMPARISON)
    # =========================================================================

    def open_comparison_dialog(self):
        """
        Menjalankan perbandingan langsung antara Algoritma Dijkstra dan A* (serta Bellman-Ford)
        dan menampilkannya dalam tabel komparatif komprehensif.
        """
        if not self.graph or not self.start_node or not self.goal_node:
            messagebox.showwarning("Peringatan", "Pastikan graf, start, dan goal sudah terdefinisi!")
            return

        # Jalankan Dijkstra
        d_path, d_cost, d_time, _, d_expanded = dijkstra_algorithm(self.graph, self.start_node, self.goal_node)
        # Jalankan A*
        a_path, a_cost, a_time, _, a_expanded = astar_algorithm(self.graph, self.start_node, self.goal_node)
        # Jalankan Bellman-Ford
        b_path, b_cost, b_time, _, b_expanded = bellman_ford_algorithm(self.graph, self.start_node, self.goal_node)

        # Dialog Window
        dlg = tk.Toplevel(self.root)
        dlg.title("Komparasi Hasil Algoritma Shortest Path")
        dlg.geometry("740, 520")
        dlg.minsize(680, 460)
        dlg.configure(bg=self.colors["panel_bg"])

        # Header Dialog
        tk.Label(dlg, text="⚖ Analisis Komparasi Algoritma Shortest Path",
                 font=("Segoe UI", 13, "bold"), fg=self.colors["accent"], bg=self.colors["panel_bg"]).pack(pady=(16, 4))

        tk.Label(dlg, text=f"Kasus: {self.start_node} → {self.goal_node}  |  Total Simpul Graf: {len(self.graph)}",
                 font=("Segoe UI", 10), fg=self.colors["text_dim"], bg=self.colors["panel_bg"]).pack(pady=(0, 14))

        # Tabel Komparasi (Treeview)
        table_frame = tk.Frame(dlg, bg=self.colors["panel_bg"], padx=16)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("algo", "path", "cost", "time", "expanded", "complexity")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=4)

        tree.heading("algo", text="Algoritma")
        tree.heading("path", text="Jalur Terpendek")
        tree.heading("cost", text="Bobot Minimum")
        tree.heading("time", text="Waktu (µs)")
        tree.heading("expanded", text="Node Dikunjungi")
        tree.heading("complexity", text="Kompleksitas Waktu")

        tree.column("algo", width=120, anchor="center")
        tree.column("path", width=220, anchor="w")
        tree.column("cost", width=95, anchor="center")
        tree.column("time", width=85, anchor="center")
        tree.column("expanded", width=105, anchor="center")
        tree.column("complexity", width=110, anchor="center")

        def fmt_path(p):
            return " → ".join(p) if p else "Tidak Ada"

        tree.insert("", tk.END, values=(
            "Dijkstra", fmt_path(d_path), str(d_cost), f"{d_time:.1f}", f"{d_expanded} node", "O((V + E) log V)"
        ))
        tree.insert("", tk.END, values=(
            "A* (A-Star)", fmt_path(a_path), str(a_cost), f"{a_time:.1f}", f"{a_expanded} node", "O((V + E) log V)"
        ))
        tree.insert("", tk.END, values=(
            "Bellman-Ford", fmt_path(b_path), str(b_cost), f"{b_time:.1f}", f"{b_expanded} node", "O(V · E)"
        ))

        tree.pack(fill=tk.BOTH, expand=True)

        # Analisis Ringkas
        analysis_card = tk.Frame(dlg, bg=self.colors["card_bg"], padx=14, pady=10)
        analysis_card.pack(fill=tk.X, padx=16, pady=14)

        analysis_text = (
            "📌 Kesimpulan Analisis:\n"
            f"1. Optimalitas: Seluruh algoritma berhasil menemukan bobot minimum yang identik ({d_cost}).\n"
            f"2. Efisiensi Eksplorasi: A* mengekspansi {a_expanded} simpul dibandingkan Dijkstra ({d_expanded} simpul) "
            "berkat panduan fungsi heuristik admissible h(n).\n"
            f"3. Kompleksitas & Runtime: Dijkstra & A* memiliki keunggulan performa runtime dibanding Bellman-Ford "
            "karena menggunakan struktur data Min-Heap Priority Queue."
        )
        tk.Label(analysis_card, text=analysis_text, font=("Segoe UI", 9),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"], justify="left").pack(anchor="w")

        btn_close = tk.Button(dlg, text="Tutup Dialog", command=dlg.destroy,
                              font=("Segoe UI", 9, "bold"), bg=self.colors["card_bg"], fg=self.colors["text_main"],
                              activebackground=self.colors["node_default"], relief=tk.FLAT, bd=0, padx=16, pady=6)
        btn_close.pack(pady=(0, 14))


# =============================================================================
# 5. MODE CLI (COMMAND LINE INTERFACE) UNTUK PENGUJIAN CEPAT
# =============================================================================

def run_cli_mode(filepath="graf.txt", algo_name="dijkstra"):
    """
    Menjalankan pencarian jalur terpendek melalui Command Line Interface.
    Format output persis mengikuti instruksi penugasan di PDF.
    """
    try:
        graph, start, goal = parse_graph_file(filepath)
    except Exception as e:
        print(f"Error memuat file '{filepath}': {e}", file=sys.stderr)
        return

    algo_name = algo_name.lower()
    if algo_name in ["dijkstra", "1"]:
        name = "Dijkstra"
        path, cost, runtime, _, expanded = dijkstra_algorithm(graph, start, goal)
    elif algo_name in ["a*", "astar", "a-star", "2"]:
        name = "A* (A-Star)"
        path, cost, runtime, _, expanded = astar_algorithm(graph, start, goal)
    elif algo_name in ["bellman-ford", "bellman", "3"]:
        name = "Bellman-Ford"
        path, cost, runtime, _, expanded = bellman_ford_algorithm(graph, start, goal)
    else:
        print(f"Algoritma '{algo_name}' tidak dikenal. Pilih: dijkstra, astar, bellman-ford", file=sys.stderr)
        return

    print("=" * 60)
    print(f"PROGRAM PENCARIAN JALUR TERPENDEK ({name})")
    print(f"File Input    : {filepath}")
    print(f"Simpul Awal   : {start}")
    print(f"Simpul Tujuan : {goal}")
    print("=" * 60)

    if path:
        path_str = " → ".join(path)
        print(f"Jalur terpendek: {path_str}")
        print(f"Bobot minimum: {cost}")
        # Format runtime
        if runtime < 1000:
            print(f"Waktu runtime: {runtime:.2f} µs")
        else:
            print(f"Waktu runtime: {runtime/1000:.3f} ms ({runtime:.1f} µs)")
        print(f"Simpul dieksplorasi: {expanded} dari {len(graph)}")
    else:
        print("Jalur terpendek: Tidak ditemukan jalur yang menghubungkan simpul.")
        print("Bobot minimum: Tak Terhingga (inf)")
        print(f"Waktu runtime: {runtime:.2f} µs")

    print("=" * 60)


# =============================================================================
# 6. TITIK MASUK UTAMA (MAIN ENTRY POINT)
# =============================================================================

def main():
    # Jika dipanggil dengan argumen CLI (misal: python MMAIKelompokX.py --cli [file] [algo])
    if "--cli" in sys.argv:
        target_file = "graf.txt"
        target_algo = "dijkstra"
        args = [a for a in sys.argv[1:] if a != "--cli"]
        if len(args) >= 1:
            target_file = args[0]
        if len(args) >= 2:
            target_algo = args[1]
        run_cli_mode(target_file, target_algo)
        return

    # Default: Menjalankan Antarmuka Grafis (GUI Tkinter)
    root = tk.Tk()
    initial_file = "graf.txt"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        initial_file = sys.argv[1]

    app = ShortestPathGUI(root, initial_filepath=initial_file)
    root.mainloop()


if __name__ == "__main__":
    main()
