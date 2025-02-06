import itertools
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Back button logic
def back_button():
    """Function to display a back button and handle navigation."""
    if st.session_state.menu != "main" and st.button("Back"):
        if st.session_state.menu == "upload_data":
            st.session_state.menu = "main"
        elif st.session_state.menu == "analysis":
            st.session_state.menu = "upload_data"
        st.rerun()

# Helper functions for loading social networks
def process_single_network(file, data_type="stock_returns"):
    """
    Process a network file. Handles stock returns correlation data and ensures non-negative edge weights.
    Cleans column names to only include company names.

    Parameters:
    - file (str): Path to the CSV file.
    - data_type (str): Defaults to "stock_returns" for this dataset.

    Returns:
    - pd.DataFrame: Processed edge list with "from", "to", and "weight" columns.
    """
    try:
        print(f"Processing file: {file} as {data_type}")

        # Load the stock returns dataset
        df = pd.read_csv(file)

        if "ret.adjusted.prices.ref.date" not in df.columns:
            raise ValueError("Dataset must contain 'ret.adjusted.prices.ref.date' as the first column.")

        # Parse the date column and set it as the index
        df["ret.adjusted.prices.ref.date"] = pd.to_datetime(df["ret.adjusted.prices.ref.date"])
        df.set_index("ret.adjusted.prices.ref.date", inplace=True)

        # Clean column names to keep only the company names
        df.columns = [col.replace("ret.adjusted.prices.", "").replace(".L", "") for col in df.columns]

        # Drop any columns that are completely empty
        df = df.dropna(axis=1, how="all")

        # Handle missing values by forward and backward filling
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Convert correlation matrix to edge list
        corr_stack = corr_matrix.stack().reset_index()
        corr_stack.columns = ["from", "to", "weight"]

        # Transform weights to non-negative values (e.g., absolute correlations)
        corr_stack["weight"] = corr_stack["weight"].abs()

        # Remove duplicate edges for undirected graph
        corr_stack["pair"] = corr_stack.apply(lambda row: tuple(sorted([row["from"], row["to"]])), axis=1)
        single_network = corr_stack.groupby("pair").agg({"weight": "mean"}).reset_index()
        single_network[["from", "to"]] = pd.DataFrame(single_network["pair"].tolist(), index=single_network.index)
        single_network.drop(columns=["pair"], inplace=True)

        # Filter out self-loops (edges where "from" == "to")
        single_network = single_network[single_network["from"] != single_network["to"]]

        return single_network

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None
      
# Function to fetch CSV from GitHub
@st.cache_data
def load_github_csv(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None
      
# Initialize session state
if "menu" not in st.session_state:
    st.session_state.menu = "main"

if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None

# Main Page
if st.session_state.menu == "main":
    st.markdown("<h1 style='text-align: center;'>REDE:Vision business edition</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>🤫 Shhhhhh... the stocks are talking</h5>", unsafe_allow_html=True)

    # Analysis selection
    analysis_type = st.selectbox(
        "Choose an analysis",
        options=["-", "Stock Market Analysis"],
        index=0
    )
    
    if analysis_type != "-":
        st.session_state.analysis_type = analysis_type

        # Ensure we only load data once
        if "G" not in st.session_state:
            github_raw_url = "https://raw.githubusercontent.com/andredspereira/rede-vision-stock-market/main/data.csv"
            edgelist = load_github_csv(github_raw_url)         
            
            if edgelist is not None:
                st.write("Processed Edgelist:")
                st.dataframe(edgelist)

                # Create NetworkX graph
                G = nx.Graph()
                for _, row in edgelist.iterrows():
                    G.add_edge(row["from"], row["to"], weight=row["weight"])

                # Save graph in session state
                st.session_state.G = G

                # Navigate to analysis
                st.session_state.menu = "analysis"
                st.rerun()
              
# Analysis Pages
if st.session_state.menu == "analysis":
    if st.session_state.analysis_type == "Stock Market Analysis":
        st.title("Stock Market Analysis")

        # Retrieve the graph from session state
        G = st.session_state.get("G", None)

        try:
            # Check for empty graph
            if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
                st.warning("The graph is empty. Please ensure the data contains connections.")
                st.stop()

            # Calculate max and min spanning trees
            tree_min = nx.minimum_spanning_tree(G)
            tree_max = nx.maximum_spanning_tree(G)

            # Compute network metrics
            degree_dict_max = dict(nx.degree(tree_max))
            degree_dict_min = dict(nx.degree(tree_min))
            eigenvector_dict = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
            betweenness_dict = nx.betweenness_centrality(G, weight="weight")
            clustering_coefficient = nx.average_clustering(G, weight="weight")
            density = nx.density(G)
            assortativity = nx.degree_assortativity_coefficient(G)
            connected_components = list(nx.connected_components(G))

            # Trending stocks (top 3 by degree centrality in max spanning tree)
            trending_stocks = sorted(degree_dict_max, key=degree_dict_max.get, reverse=True)[:3]
            trending_text = ", ".join(f"**{stock}**" for stock in trending_stocks)

            # Safest stocks (top 3 by degree centrality in min spanning tree)
            safest_stocks = sorted(degree_dict_min, key=degree_dict_min.get, reverse=True)[:3]
            safest_text = ", ".join(f"**{stock}**" for stock in safest_stocks)

            # Critical industry stocks (top 3 by betweenness centrality)
            critical_stocks = sorted(betweenness_dict, key=betweenness_dict.get, reverse=True)[:3]
            critical_text = ", ".join(f"**{stock}**" for stock in critical_stocks)

            # Market cohesion insight
            if clustering_coefficient < 0.3:
                clustering_text = "has low connectedness, and if a company in the network crashes or goes bankrupt, its neighbors will be very affected"
            elif 0.3 <= clustering_coefficient <= 0.6:
                clustering_text = "has average connectedness, and the companies in the same field of the affected company will be impacted."
            else:
                clustering_text = "is very well connected, so there should be no major issues to other stocks if a company goes bankrupt for external reasons."

            # Assortativity insight
            if assortativity < 0.1:
                assortativity_text = (
                    "The market is a free-scale network as the stock market prices of the biggest companies are connected to many smaller companies. "
                    "If a large company is doing well, it is best to invest in the companies it is connected to in the graph."
                )
            else:
                assortativity_text = (
                    "The market is structured in a way that the major companies are mostly connected together. "
                    "As such, investing in multiple major companies is risky, and it is best to look for smaller companies to invest in for now."
                )

            # Dropdown to select metrics or nodes
            options = ["Insights Summary", "Stock insight"] 
            selected_option = st.selectbox("Choose insight level", options, index=0)

            # Insights Summary
            if selected_option == "Insights Summary":
                # Display insights
                st.subheader("Insights Summary")
                st.markdown(f"""
                - **Trending Stocks**: The stocks that follow the trend of the market the most are: {trending_text}.
                - **Safest Stocks**: The stocks that can be invested in for safer investments in the case of a market crash are: {safest_text}.
                - **Critical Industry Stocks**: If a shift happens in the market, the stocks that will be key indicators of affected sectors are: {critical_text}.
                - **Clustering Coefficient**: The average clustering coefficient is **{clustering_coefficient:.2f}**, indicating that the network {clustering_text}.
                - **Assortativity**: {assortativity_text}
                """)

            # Node-specific Metrics 
            else:
                stock_options = list(G.nodes())  # List of all stocks (nodes)
                selected_stock = st.selectbox("Select a stock to analyze", stock_options)
                if selected_stock:
                  stock = selected_stock
                  # Compute node-specific metrics
                  degree = degree_dict_max.get(stock, 0)  # Degree centrality in max spanning tree
                  safety_degree = degree_dict_min.get(stock, 0)  # Degree centrality in min spanning tree
                  eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight", max_iter=1000).get(stock, 0)
                  betweenness = betweenness_dict.get(stock, 0)
                  closeness = nx.closeness_centrality(G, distance="weight").get(stock, 0)
                  neighbors = list(G.neighbors(stock))
                  most_connected_neighbor = max(neighbors, key=lambda x: G[stock][x]["weight"], default=None)
  
                  # Insights based on node-specific metrics
                  trending_insight = f"**{stock}** is {'a major' if degree >= 3 else 'a minor'} trend follower."
                  safety_insight = f"This stock is {'a relatively safe' if safety_degree >= 3 else 'a less safe'} investment in a market crash."
                  critical_insight = f"**{stock}** {'is a critical connector in the market network' if betweenness > 0.1 else 'plays a less central role in the connectivity of the market.'}"
                  eigenvector_insight = f"**{stock}** is {'a highly influential stock in the market, strongly connected to other key players' if eigenvector_centrality > 0.3 else 'less influential in the market, with weaker connections to prominent stocks.'}"
                  closeness_insight = f"**{stock}** {'has a strategic position, with quick access to all parts of the network, indicating high market agility' if closeness > 0.5 else 'is less centrally positioned, with longer distances to other stocks in the network, which may reduce its market agility.'}"
  
                  st.subheader(f"Stock Insights: {stock}")
                  st.markdown(f"""
                  - **Trending Insight**: {trending_insight}
                  - **Safety Insight**: {safety_insight}
                  - **Criticality Insight**: {critical_insight}
                  - **Eigenvector Centrality**: {eigenvector_insight}
                  - **Closeness Centrality**: {closeness_insight}
                  - **Most Connected Neighbor**: {most_connected_neighbor or 'None'}, with the strongest connection weight.
                  """)

            # Visualization of spanning trees
            def draw_tree(tree, title, selected_node=None):
                fig, ax = plt.subplots(figsize=(10, 6))
                layout = nx.spring_layout(tree, seed=42)

                # Default node colors and outlines
                node_colors = ["ivory"] * len(tree.nodes())
                node_outlines = ["black"] * len(tree.nodes())
                selected_node_color = "skyblue"
                neighbor_node_color = "lightgreen"

                # Highlight the selected node and its neighbors
                if selected_node in tree.nodes:
                    neighbors = list(tree.neighbors(selected_node))
                    for i, node in enumerate(tree.nodes()):
                        if node == selected_node:
                            node_colors[i] = selected_node_color  # Selected node
                        elif node in neighbors:
                            node_colors[i] = neighbor_node_color  # Neighbor nodes

                # Normalize edge weights for consistent thickness
                weights = [tree[u][v]["weight"] for u, v in tree.edges()]
                min_weight, max_weight = min(weights), max(weights)
                normalized_weights = [0.5 + 2.5 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]

                # Draw the tree with customized styles
                nx.draw(
                    tree,
                    pos=layout,
                    ax=ax,
                    with_labels=True,
                    node_size=700,
                    node_color=node_colors,
                    font_size=10,
                    width=normalized_weights,
                    edge_color="gray",
                    edgecolors=node_outlines,  # Add black outlines to nodes
                )
                ax.set_title(title)

                # Add legend
                legend_elements = [
                    plt.Line2D([0], [0], marker="o", color="w", label="Selected Node", markerfacecolor=selected_node_color, markeredgecolor="black"),
                    plt.Line2D([0], [0], marker="o", color="w", label="Neighbor Node", markerfacecolor=neighbor_node_color, markeredgecolor="black"),
                ]
                ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

                st.pyplot(fig)

            # Display spanning tree visualizations
            st.subheader("Spanning Trees")

            # Minimum Spanning Tree
            st.write("### Minimum Spanning Tree")
            draw_tree(tree_min, "Minimum Spanning Tree", selected_node=selected_option)

            # Maximum Spanning Tree
            st.write("### Maximum Spanning Tree")
            draw_tree(tree_max, "Maximum Spanning Tree", selected_node=selected_option)
        except Exception as e:
            st.error(f"Error calculating network metrics: {e}")

        back_button()  # Add the back button

    elif st.session_state.analysis_type == "Marketing Analysis":
        st.title("Marketing Analysis")

        # Retrieve the graph from session state
        G = st.session_state.get("G", None)

        try:
            # Check for empty graph
            if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
                st.warning("The graph is empty. Please ensure the data contains connections.")
                st.stop()

            # Compute network metrics
            degree_dict = dict(nx.degree(G))
            strength_dict = {node: sum(data["weight"] for _, _, data in G.edges(node, data=True)) for node in G}
            eigenvector_dict = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
            betweenness_dict = nx.betweenness_centrality(G, weight="weight")
            clustering_coefficient = nx.average_clustering(G, weight="weight")
            density = nx.density(G)
            connected_components = list(nx.connected_components(G))

            # Closeness centrality: handle disconnected nodes
            closeness_dict = {}
            for component in connected_components:
                subgraph = G.subgraph(component)
                closeness_partial = nx.closeness_centrality(subgraph, distance="weight")
                closeness_dict.update(closeness_partial)
            for node in G.nodes():
                if node not in closeness_dict:
                    closeness_dict[node] = 0

            # Dropdown to select metrics or nodes
            options = ["Insights Summary", "Metrics Overview"] + list(G.nodes())
            selected_option = st.selectbox("Select metrics or customer to view", options, index=0)

            # Insights Summary
            if selected_option == "Insights Summary":
                st.subheader("Insights Summary")

                # Key Customers by Eigenvector Centrality (influence)
                influential_customers = sorted(eigenvector_dict, key=eigenvector_dict.get, reverse=True)[:3]
                influential_text = ", ".join(f"**{customer}**" for customer in influential_customers)

                # Key Customers by Betweenness Centrality (critical connectors)
                critical_customers = sorted(betweenness_dict, key=betweenness_dict.get, reverse=True)[:3]
                critical_text = ", ".join(f"**{customer}**" for customer in critical_customers)

                # Market Cohesion
                cohesion_status = "highly cohesive" if density > 0.5 else "moderately cohesive"
                components_text = f"The network consists of **{len(connected_components)}** connected component(s)."

                st.markdown(f"""
                - **Influential Customers**: The customers with the most influence in the network are: {influential_text}.
                - **Key Connectors**: The customers acting as key connectors are: {critical_text}.
                - **Market Cohesion**: The network is **{cohesion_status}**, with a density of **{density:.2f}**.
                - **Connected Components**: {components_text}.
                - **Clustering Coefficient**: The average clustering coefficient is **{clustering_coefficient:.2f}**, indicating the level of interconnectedness among customers.
                """)

            # Metrics Overview
            elif selected_option == "Metrics Overview":
                metrics = [
                    {"Metric": "Density", "Value": round(density, 2)},
                    {"Metric": "Clustering Coefficient", "Value": round(clustering_coefficient, 2)},
                    {"Metric": "Connected Components", "Value": len(connected_components)},
                    {"Metric": "Top Customer by Eigenvector Centrality", "Value": max(eigenvector_dict, key=eigenvector_dict.get)},
                    {"Metric": "Top Customer by Betweenness Centrality", "Value": max(betweenness_dict, key=betweenness_dict.get)},
                ]
                st.table(pd.DataFrame(metrics))

            # Node-specific Metrics
            else:
                customer = selected_option
                neighbors = list(G.neighbors(customer))
                most_connected = max(neighbors, key=lambda x: G[customer][x]["weight"], default=None)
                metrics = [
                    {"Metric": "Degree (Number of Connections)", "Value": degree_dict[customer]},
                    {"Metric": "Strength (Total Weight of Connections)", "Value": strength_dict[customer]},
                    {"Metric": "Eigenvector Centrality", "Value": round(eigenvector_dict[customer], 2)},
                    {"Metric": "Closeness Centrality", "Value": round(closeness_dict[customer], 2)},
                    {"Metric": "Betweenness Centrality", "Value": round(betweenness_dict[customer], 2)},
                    {"Metric": "Most Connected Customer", "Value": most_connected},
                ]
                st.table(pd.DataFrame(metrics))

            # Network Visualization
            st.write("### Network Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Network visualization setup
            layout = nx.spring_layout(G, seed=42, weight=None)  # Fixed-length edges by ignoring weights
            node_colors = ["ivory"] * len(G.nodes())  # Default color for all nodes
            node_outlines = ["black"] * len(G.nodes())  # Default outline for all nodes

            # Variables for legend
            selected_node_color = "skyblue"
            neighbor_node_color = "lightgreen"
            default_node_color = "ivory"

            # Highlight selected node and its neighbors
            if selected_option in G.nodes():
                customer = selected_option
                neighbors = list(G.neighbors(customer))
                for i, node in enumerate(G.nodes()):
                    if node == customer:
                        node_colors[i] = selected_node_color  # Highlight selected customer
                        node_outlines[i] = "black"
                    elif node in neighbors:
                        node_colors[i] = neighbor_node_color  # Highlight neighbors
                        node_outlines[i] = "black"

            # Normalize edge weights for thickness
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            min_weight, max_weight = min(weights), max(weights)
            normalized_weights = [
                0.5 + 2.5 * ((w - min_weight) / (max_weight - min_weight)) for w in weights
            ]

            # Draw the graph
            nx.draw(
                G,
                pos=layout,
                ax=ax,
                with_labels=True,
                node_size=700,
                node_color=node_colors,
                font_size=10,
                width=normalized_weights,
                edge_color="gray",
                edgecolors=node_outlines,  # Ensure black outlines
            )

            # Add dynamic legend
            legend_elements = [
                plt.Line2D([0], [0], marker="o", color="w", label="Selected Customer", markerfacecolor=selected_node_color, markeredgecolor="black"),
                plt.Line2D([0], [0], marker="o", color="w", label="Neighbor Customer", markerfacecolor=neighbor_node_color, markeredgecolor="black"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error calculating network metrics: {e}")
        
        back_button()  # Add the back button

