import json
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from PIL import Image

def plot_scene_graph(
    scene_graph: dict,
    image_path: str = None,
    output_path: str = "scene_graph_dark_side_by_side.png",
    title: str = "Grounded Visual Scene Graph"
):
    """
    Renders a high-contrast dark-themed side-by-side visualization:
    Left: Input image with color-coded bounding boxes.
    Right: Directed scene graph with high-contrast node and edge badges.
    """
    # Dark theme color palette
    BG_COLOR = "#0F172A"       # Slate 900
    PANEL_BG = "#1E293B"       # Slate 800
    TEXT_LIGHT = "#F8FAFC"     # Slate 50
    EDGE_COLOR = "#94A3B8"     # Slate 400
    EDGE_BADGE_BG = "#E11D48"  # Rose 600
    NODE_BORDER = "#818CF8"    # Indigo 400
    NODE_BG = "#312E81"        # Indigo 900
    
    PALETTE = ["#38BDF8", "#F472B6", "#4ADE80", "#FBBF24", "#A78BFA"]

    G = nx.DiGraph()
    id_to_label = {}
    
    nodes = scene_graph.get("nodes", [])
    edges = scene_graph.get("edges", [])

    for i, node in enumerate(nodes):
        nid = node.get("id") or node.get("name")
        name = node.get("name", nid)
        attrs = node.get("attributes", [])
        
        # Format clean wrapped label with attributes
        wrapped_name = textwrap.fill(name.title(), width=16)
        if attrs:
            attr_str = textwrap.fill(" • ".join(attrs[:2]), width=20)
            label = f"{wrapped_name}\n[{attr_str}]"
        else:
            label = wrapped_name
            
        id_to_label[nid] = label
        id_to_label[node.get("name")] = label
        G.add_node(label, name=name, color=PALETTE[i % len(PALETTE)])

    for edge in edges:
        subj = id_to_label.get(edge.get("subject"), edge.get("subject"))
        obj = id_to_label.get(edge.get("object"), edge.get("object"))
        pred = edge.get("predicate", "related_to").upper()
        if subj and obj and subj != obj:
            G.add_edge(subj, obj, label=pred)

    # Setup dark figure
    fig = plt.figure(figsize=(18, 8.5), dpi=180, facecolor=BG_COLOR)
    
    if image_path:
        ax_img = plt.subplot(1, 2, 1)
        ax_graph = plt.subplot(1, 2, 2)
        
        # Left Panel: Clean original image without bounding boxes
        ax_img.set_facecolor(PANEL_BG)
        img = Image.open(image_path).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_title("Input Image", color=TEXT_LIGHT, fontsize=13, weight="bold", pad=12)
        ax_img.axis("off")
        ax_draw = ax_graph
    else:
        ax_draw = plt.subplot(1, 1, 1)

    # Right Panel: Dark themed Graph
    ax_draw.set_facecolor(PANEL_BG)
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=60)
    
    # Draw edges with prominent arrows
    nx.draw_networkx_edges(
        G, pos, ax=ax_draw,
        edge_color=EDGE_COLOR,
        arrowsize=24,
        arrowstyle="-|>",
        width=2.5,
        connectionstyle="arc3,rad=0.15",
        alpha=0.85
    )
    
    # Draw node text badges with high contrast borders
    for node, (x, y) in pos.items():
        node_color = G.nodes[node].get("color", NODE_BORDER)
        ax_draw.text(
            x, y, node,
            color=TEXT_LIGHT,
            fontsize=9.5,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.55,rounding_size=0.3",
                facecolor=NODE_BG,
                edgecolor=node_color,
                linewidth=2.5,
                alpha=0.96
            )
        )

    # Draw edge relation badges with bright rose background
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        ax=ax_draw,
        font_color=TEXT_LIGHT,
        font_size=9.5,
        font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4,rounding_size=0.2",
            facecolor=EDGE_BADGE_BG,
            edgecolor=TEXT_LIGHT,
            linewidth=1.2,
            alpha=0.95
        )
    )

    ax_draw.set_title(title, color=TEXT_LIGHT, fontsize=13, weight="bold", pad=12)
    ax_draw.axis("off")
    
    # Adjust margins
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04, wspace=0.08)
    plt.savefig(output_path, facecolor=BG_COLOR, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved dark side-by-side visualization to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Grounded Scene Graph (Dark Theme)")
    parser.add_argument("--image", type=str, default="images/29.jpg", help="Path to source image")
    parser.add_argument("--index", type=int, default=1, help="Index of benchmark query to visualize (0-23)")
    parser.add_argument("--output", type=str, default="scene_graph_dark_side_by_side.png", help="Output file path")
    args = parser.parse_args()

    benchmark_data = json.load(open("kaggle_run_24/output/benchmark_24_results.json", encoding="utf-8"))["results"]
    sample = benchmark_data[args.index]
    sg = sample["parsed_json"]["grounded_scene_graph"]
    img_path = args.image or f"images/{sample['image']}"

    plot_scene_graph(
        scene_graph=sg,
        image_path=img_path,
        output_path=args.output,
        title=f"Query: {sample['question']}"
    )
