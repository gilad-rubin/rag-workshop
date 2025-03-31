import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.manifold import TSNE


def nicer_print(text, width=80):
    """
    Print text with a width of 80 characters
    """
    print(textwrap.fill(text, width=width))


from matplotlib import rcParams


def visualize_document_embeddings(
    documents, color_by="page_number", figsize=(12, 10), text_truncate=100
):
    """
    Create an interactive 2D projection of document embeddings using t-SNE with hover functionality.

    Args:
        documents: List of documents with embeddings
        color_by: Metadata key to use for coloring points (default: "page_number")
        figsize: Size of the figure as (width, height) tuple
        text_truncate: Maximum number of characters to show in hover text

    Returns:
        The t-SNE projection for further use if needed
    """
    # Set visualization style
    plt.style.use("seaborn-v0_8-whitegrid")
    rcParams["figure.figsize"] = figsize

    # Extract embeddings
    embeddings = np.array([doc.embedding for doc in documents])

    # Perform dimensionality reduction
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) - 1),
        learning_rate="auto",
        init="pca",
    )
    projection = tsne.fit_transform(embeddings)

    # Get color values from meta field
    color_values = extract_meta_values(documents, color_by)

    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=figsize)

    # Create a scatter plot with more vibrant colors
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=color_values,
        cmap="viridis",
        alpha=0.8,
        s=90,
        edgecolor="none",
        picker=True,  # Enable picking for hover events
    )

    # Add a horizontal color bar
    if len(set(color_values)) > 1:
        cbar = plt.colorbar(
            scatter, label=color_by, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cbar.ax.tick_params(labelsize=9)

    # Set title and labels with clean styling
    ax.set_title(
        f"2D t-SNE Projection of Document Embeddings (colored by {color_by})",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    # Cleaner grid with lighter lines
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    # Create annotation object (initially hidden)
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    # Store document info for hover
    point_info = []
    for i, doc in enumerate(documents):
        # Truncate content text
        content = doc.content if hasattr(doc, "content") else "No content"
        truncated_text = content[:text_truncate] + (
            "..." if len(content) > text_truncate else ""
        )

        # Format metadata
        meta_str = ""
        if hasattr(doc, "meta") and doc.meta:
            meta_str = "\n".join(
                [
                    f"{k}: {v}"
                    for k, v in doc.meta.items()
                    if k in [color_by, "page_number", "file_path", "split_id"]
                ]
            )

        point_info.append(f"Document {i}\n{meta_str}\n\nContent:\n{truncated_text}")

    def hover(event):
        # Check if the mouse is over a point
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                # Get the index of the hovered point
                index = ind["ind"][0]

                # Update annotation with document info
                annot.xy = (projection[index, 0], projection[index, 1])
                annot.set_text(point_info[index])
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    # Connect hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.tight_layout()
    plt.show()

    return projection


def extract_meta_values(documents, meta_key):
    """
    Extract values from the meta field of documents.

    Args:
        documents: List of document objects
        meta_key: Key in the meta dictionary to extract values from

    Returns:
        List of numeric values for coloring
    """
    # Extract values from meta field
    try:
        values = [doc.meta.get(meta_key, 0) for doc in documents]

        # Handle non-numeric values if needed
        if not all(isinstance(val, (int, float)) for val in values):
            unique_values = sorted(set(values))
            value_map = {val: i for i, val in enumerate(unique_values)}
            values = [value_map[val] for val in values]

        return values
    except (AttributeError, KeyError):
        # Fallback to default coloring if meta key doesn't exist
        return np.zeros(len(documents))


def display_pipeline(pipeline):
    """
    Display a Haystack pipeline as a minimal Mermaid diagram.

    Args:
        pipeline: A Haystack Pipeline object

    Returns:
        None: Displays the Mermaid diagram in the notebook
    """

    # Get the pipeline graph
    graph = pipeline.graph

    # Prepare mermaid syntax with graph configuration
    mermaid_code = [
        "```mermaid",
        "%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'arial', 'lineWidth': '2px', 'edgeLabelBackground': '#ffffff'}}}%%",
        "graph TD;",
    ]

    # Define nodes (components) with improved styling
    for node, data in graph.nodes(data=True):
        if node not in ["input", "output"]:
            component_type = type(data["instance"]).__name__
            mermaid_code.append(
                f'    {node}["{node}<br><small><i>{component_type}</i></small>"]'
            )

    # Define connections with improved styling
    for from_comp, to_comp, data in graph.edges(data=True):
        if from_comp not in ["input", "output"] and to_comp not in ["input", "output"]:
            # Arrow connection with thicker line
            mermaid_code.append(f"    {from_comp} ==> {to_comp}")

    # Add explicit input connections with styling
    input_connections = {
        (_, to_comp) for _, to_comp in graph.out_edges("input") if to_comp != "output"
    }
    for _, to_comp in input_connections:
        mermaid_code.append(f'    input(("Input")) ==> {to_comp}')

    # Add explicit output connections with styling
    output_connections = {
        (from_comp, _)
        for from_comp, _ in graph.in_edges("output")
        if from_comp != "input"
    }
    for from_comp, _ in output_connections:
        mermaid_code.append(f'    {from_comp} ==> output(("Output"))')

    mermaid_code.append("```")

    # Display in notebook with added spacing
    display(Markdown("\n".join(mermaid_code)))


pd.set_option("display.max_colwidth", 100)


def visualize_document_scores(documents, metadata_field="file_path"):
    """
    Visualize the relevance scores of retrieved documents grouped by metadata field.
    Each group is displayed as a bucket with individual bars for each document.
    Groups are sorted by their document count (descending), then by maximum score.
    Documents within each group are sorted by score.

    Args:
        documents: List of retrieved documents with score and metadata attributes
        metadata_field: The metadata field to group by (default: "file_path")
    """
    # Extract scores and metadata
    scores = [doc.score for doc in documents]
    metadata_values = [
        doc.meta.get(metadata_field, "Unknown") if hasattr(doc, "meta") else "Unknown"
        for doc in documents
    ]

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({"score": scores, metadata_field: metadata_values})

    # Get just the filename from the path for cleaner display if metadata is file_path
    if metadata_field == "file_path":
        df["display_name"] = df[metadata_field].apply(
            lambda x: x.split("/")[-1] if isinstance(x, str) else "Unknown"
        )
    else:
        df["display_name"] = df[metadata_field]

    # Calculate max score and count per group for sorting
    group_stats = df.groupby(metadata_field).agg(
        group_max_score=("score", "max"), group_count=("score", "count")
    )

    # Sort by count (descending), then by max score (descending)
    group_stats = group_stats.sort_values(
        ["group_count", "group_max_score"], ascending=[False, False]
    )

    # Map the group stats back to the dataframe
    df["group_max_score"] = df[metadata_field].map(group_stats["group_max_score"])
    df["group_count"] = df[metadata_field].map(group_stats["group_count"])

    # Sort the dataframe: first by group count (descending), then by group's max score (descending),
    # then by individual score within group (descending)
    df = df.sort_values(
        ["group_count", "group_max_score", metadata_field, "score"],
        ascending=[False, False, True, False],
    )

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Prepare for plotting
    groups = df[metadata_field].unique()
    n_groups = len(groups)
    group_positions = np.arange(n_groups)

    # Map groups to positions based on the new sorting
    group_to_position = {group: i for i, group in enumerate(groups)}

    # Create a distinct color palette for bars within each group
    color_map = plt.cm.tab10  # Using tab10 for more distinct colors

    # Fixed width for all bars
    width = 0.2

    # Plot individual bars for each document
    for idx, (_, row) in enumerate(df.iterrows()):
        group_pos = group_to_position[row[metadata_field]]
        # Calculate position within group based on document count
        group_docs = df[df[metadata_field] == row[metadata_field]]
        # Find position of current document within its group
        doc_position = group_docs.index.get_indexer([row.name])[0]

        # Calculate offset to center the bars within each group
        total_width = width * len(group_docs)
        start_offset = -total_width / 2
        offset = start_offset + (doc_position * width) + (width / 2)

        # Use distinct colors for each document within a group
        color_idx = doc_position % 10  # tab10 has 10 distinct colors

        # Plot the bar
        bar = plt.bar(
            group_pos + offset,
            row["score"],
            width=width,
            color=color_map(color_idx),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add score text on top of each bar
        plt.text(
            group_pos + offset,
            row["score"] + 0.01,  # Slightly above the bar
            f"{row['score']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    # Set the x-axis labels to the group names
    plt.xticks(group_positions, groups)
    plt.xlabel(metadata_field)
    plt.ylabel("Relevance Score")
    plt.title(f"Document Relevance Scores Grouped by {metadata_field}")
    plt.ylim(0, df["score"].max() * 1.2)  # Add more padding to accommodate score labels
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Display detailed scores within each group
    print(f"\nDetailed scores by {metadata_field}:")
    return df.sort_values(
        ["group_count", "group_max_score", metadata_field, "score"],
        ascending=[False, False, True, False],
    )
