# Create a 2D projection of document embeddings using t-SNE for visualization
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.manifold import TSNE


def nicer_print(text, width=80):
    """
    Print text with a width of 80 characters
    """
    print(textwrap.fill(text, width=width))


def create_document_embedding_visualization(documents):
    """
    Create a 2D projection of document embeddings using t-SNE and visualize them.

    Args:
        documents: List of documents with embeddings

    Returns:
        The t-SNE projection for further use if needed
    """
    # Extract embeddings from documents
    embeddings = np.array([doc.embedding for doc in documents])

    # Create a 2D t-SNE projection
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
    )
    projection = tsne.fit_transform(embeddings)

    # Visualize the projection
    plt.figure(figsize=(10, 8))
    plt.scatter(projection[:, 0], projection[:, 1], alpha=0.7)

    # Add document indices as labels
    for i, (x, y) in enumerate(projection):
        plt.annotate(str(i), (x, y), fontsize=9, alpha=0.8)

    plt.title("2D t-SNE Projection of Document Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

    return projection


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
        "%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'arial', 'lineWidth': '2px' }}}%%",
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
