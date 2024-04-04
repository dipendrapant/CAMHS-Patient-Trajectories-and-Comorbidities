####################
# For creating F90 co-occurring graphs:
# 1. Find all the columns that starts with F90 in the dataframe
# 2. List them in a list as string
# 3. Create a column for F90 diagnosis and count the number of patients that have both F90 and the other diagnosis
#################### 
# These graphs are made in 3 levels within 4 implementations: 
# 1- original level, use and show diseases in the highest detailed level of ICD-10 hierarchy and as they are in the EHR without any manipulation.
# 2- Adjusted original level,Similar to the original level, but with some practical adjustments such as node colors and sizes to highlight differences in occurrence.
# 3- Level 3, use and show diseases in the level 3 of ICD-10 hierarchy which is limitted to just 3 charachters
# 4- Level 1, use and show diseases in the lowest detailed level of ICD-10 hierarchy which is limitted to just 1 charachter of each disease
#################### 
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

load_dotenv()
######################################
# 1- Original level - without color size adjustemnets
######################################
class CoocurringDiagnosisNetworkGraph:
    def load_plot_network_graph(self, original_df):
        # Load your CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df")
        )
        # Select only diagnosis codes which are from 1 to 1306 , from 1306 onwards are ATC codes
        dummied_df = dummied_df.iloc[:, :1306]

        # Make a new dataframe containing only those rows, based on 'episode_id' in original_df
        processed_df = dummied_df[
            dummied_df["episode_id"].isin(original_df["episode_id"])
        ]

        # Merge dataframe with original_df by selecting pasient from original_df
        df = pd.merge(
            original_df[
                [
                    "pasient",
                    "episode_id",
                ]
            ],
            processed_df,
            on="episode_id",
        )

        # From the df dataframe find all the column names that starts with F90 and list them in a list as string
        F90 = [col for col in df.columns if col.startswith("F90")]

        # Create a column for F90 diagnosis
        df["F90"] = df[F90].max(axis=1)

        # Drop the original F90 columns
        df = df.drop(columns=F90)

        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])
        for diagnosis in df.columns[2:]:
            count = df[(df["F90"] == 1) & (df[diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                # print(f"F90 + {diagnosis}: {count}")
                new_row = pd.DataFrame(
                    {"F90Cooccuring": [diagnosis], "F90CooccuringValue": [count]}
                )
                df_coocurring = pd.concat([df_coocurring, new_row], ignore_index=True)

        # Check the count of F90CooccuringValue
        F90_name = df_coocurring[df_coocurring["F90Cooccuring"] == "F90"][
            "F90Cooccuring"
        ].values
        F90_count = int(
            df_coocurring[df_coocurring["F90Cooccuring"] == "F90"][
                "F90CooccuringValue"
            ].values
        )

        # Find the row in df_coocurring dataframe with the F90Cooccuring as F90 & exclude it from the df_coocurring dataframe
        df_coocurring = df_coocurring[df_coocurring["F90Cooccuring"] != "F90"]


        # Make a set variable named data from df_coocurring dataframe that will have F90Cooccuring column as key and F90CooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)
        return self.calculate_and_draw_graph(G, data, F90_count)

    def calculate_and_draw_graph(self, G, data, F90_count):
        # Sort nodes by weight in descending order
        sorted_nodes = sorted(data.items(), key=lambda x: x[1], reverse=True)

        # Calculate positions
        node_positions = {}
        angle_gap = 2 * np.pi / len(sorted_nodes)  # Gap between nodes
        for i, (node, weight) in enumerate(sorted_nodes):
            angle = i * angle_gap
            # Position nodes in a circle around (0, 0)
            # Adjust the radius or other parameters as needed
            x = np.cos(angle) * 10  # Radius of 10, adjust as needed
            y = np.sin(angle) * 10
            node_positions[node] = (x, y)

        # Ensure "F90" is at the center
        node_positions["F90"] = (0, 0)

        # Draw the graph using calculated positions
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_edges(G, node_positions, edge_color="grey")
        nx.draw_networkx_labels(
            G,
            node_positions,
            font_size=8,
            labels={node: node for node in G.nodes() if node != "F90"},
        )

        # Draw the "F90" node with a specific size
        nx.draw_networkx_nodes(
            G, node_positions, nodelist=["F90"], node_size=(F90_count * 50), node_color="orange"
        )

        # Draw other nodes with the default size
        nx.draw_networkx_nodes(
            G,
            node_positions,
            nodelist=[node for node in G.nodes() if node != "F90"],
            node_size=500,
            node_color="#60c3dd",
        )
        nx.draw_networkx_labels(G, node_positions, labels={"F90": "F90"}, font_size=20)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
        plt.axis('off')
        return plt
 
       
######################################
# 2- Adjusted Original level
######################################
class CoocurringDiagnosisNetworkGraph_adjusted:
    def load_plot_network_graph(self, original_df):
        # Load your CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df")
        )
        # Select only diagnosis codes which are from 1 to 1306 , from 1306 onwards are ATC codes
        dummied_df = dummied_df.iloc[:, :1306]

        # Make a new dataframe containing only those rows, based on 'episode_id' in original_df
        processed_df = dummied_df[
            dummied_df["episode_id"].isin(original_df["episode_id"])
        ]

        # Merge dataframe with original_df by selecting pasient from original_df
        df = pd.merge(
            original_df[
                [
                    "pasient",
                    "episode_id",
                ]
            ],
            processed_df,
            on="episode_id",
        )

        # From the df dataframe find all the column names that starts with F90 and list them in a list as string
        F90 = [col for col in df.columns if col.startswith("F90")]

        # Create a column for F90 diagnosis
        df["F90"] = df[F90].max(axis=1)

        # Drop the original F90 columns
        df = df.drop(columns=F90)

        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])
        for diagnosis in df.columns[2:]:
            count = df[(df["F90"] == 1) & (df[diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                # print(f"F90 + {diagnosis}: {count}")
                new_row = pd.DataFrame(
                    {"F90Cooccuring": [diagnosis], "F90CooccuringValue": [count]}
                )
                df_coocurring = pd.concat([df_coocurring, new_row], ignore_index=True)

        # Check the count of F90CooccuringValue
        F90_name = df_coocurring[df_coocurring["F90Cooccuring"] == "F90"][
            "F90Cooccuring"
        ].values
        F90_count = int(
            df_coocurring[df_coocurring["F90Cooccuring"] == "F90"][
                "F90CooccuringValue"
            ].values
        )

        # Find the row in df_coocurring dataframe with the F90Cooccuring as F90 & exclude it from the df_coocurring dataframe
        df_coocurring = df_coocurring[df_coocurring["F90Cooccuring"] != "F90"]
        print(df_coocurring[['F90Cooccuring', 'F90CooccuringValue']].sort_values(by='F90CooccuringValue', ascending=False))

        # Make a set variable named data from df_coocurring dataframe that will have F90Cooccuring column as key and F90CooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)
        return self.calculate_and_draw_graph(G, data, F90_count)

    def calculate_and_draw_graph(self, G, data, F90_count):
        base_size = 500  # Base size for all nodes
        scale_factor = 50  # Scale factor for node sizes based on occurrences

        # Node sizes based on co-occurrence value
        node_sizes = {node: base_size + scale_factor * weight for node, weight in data.items()}
        node_sizes["F90"] = base_size + scale_factor * F90_count

        # Separate color for the central node "F90"
        central_node_color = "darkorange"

        # Exclude "F90" from the min and max calculations for normalization
        min_size = min(size for node, size in node_sizes.items() if node != "F90")
        max_size = max(size for node, size in node_sizes.items() if node != "F90")

        # Generate a continuous colormap for other nodes
        colormap = plt.cm.viridis
        normalize = mcolors.Normalize(vmin=min_size, vmax=max_size)
        scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)

        # Assign colors to nodes, excluding "F90" from affecting the color scale
        node_colors = {node: central_node_color if node == "F90" else scalar_map.to_rgba(size)
                    for node, size in node_sizes.items()}

        pos = nx.spring_layout(G)  # Position nodes using the spring layout

        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[n] for n in G.nodes()], 
                            node_color=[node_colors[n] for n in G.nodes()], alpha=0.6)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)  # Thicker edge lines

        # Drawing edge labels with custom font color and no background
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.3, font_size=10)

        nx.draw_networkx_labels(G, pos, font_size=12)
        plt.axis('off')
        return plt

########################################################
# 3- Level 3 - adjusted
########################################################
class CoocurringDiagnosisNetworkGraph_higherlevel31:
    def load_plot_network_graph(self, original_df):
        # Load CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df")
        )
        # Select only diagnosis codes which are from 1 to 1306 , from 1306 onwards are ATC codes
        dummied_df = dummied_df.iloc[:, :1306]

        # Make a new dataframe containing only those rows, based on 'episode_id' in original_df
        processed_df = dummied_df[
            dummied_df["episode_id"].isin(original_df["episode_id"])
        ]

        # Merge dataframe with original_df by selecting pasient from original_df
        df = pd.merge(
            original_df[
                [
                    "pasient",
                    "episode_id",
                ]
            ],
            processed_df,
            on="episode_id",
        )

        # From the df dataframe find all the column names that starts with F90 and list them in a list as string
        F90 = [col for col in df.columns if col.startswith("F90")]

        # Create a column for F90 diagnosis
        df["F90"] = df[F90].max(axis=1)

        # Drop the original F90 columns
        df = df.drop(columns=F90)

        # Step 1: Map each detailed diagnosis to a high-level category
   
        # Process diagnosis columns
        for diagnosis in df.columns[2:]:
            high_level_diagnosis = diagnosis[:3]  # Get the first 3 characters

            # Identify all columns that belong to this high-level diagnosis
            related_columns = [col for col in df.columns if col.startswith(high_level_diagnosis)]
            
            # Recalculate the high-level diagnosis column by taking the max value across related detailed diagnoses
            df[high_level_diagnosis] = df[related_columns].max(axis=1)

        unique_high_level_diagnoses = set([col[:3] for col in df.columns[2:]])

        # Ensure df_coocurring is a DataFrame with the intended structure
        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])

        # Accumulate new rows in a list for efficient DataFrame construction
        new_rows = []

        for high_level_diagnosis in unique_high_level_diagnoses:
            # Count the unique patients with both the "F90" condition and the current high-level diagnosis
            count = df[(df["F90"] == 1) & (df[high_level_diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                new_row = {"F90Cooccuring": high_level_diagnosis, "F90CooccuringValue": count}
                new_rows.append(new_row)

        # Append all new rows at once to the df_coocurring DataFrame
        df_coocurring = pd.concat([df_coocurring, pd.DataFrame(new_rows)], ignore_index=True)

        # Check the count of F90CooccuringValue
        F90_count = int(
            df_coocurring[df_coocurring["F90Cooccuring"] == "F90"][
                "F90CooccuringValue"
            ].values
        )

        # Find the row in df_coocurring dataframe with the F90Cooccuring as F90 & exclude it from the df_coocurring dataframe
        df_coocurring = df_coocurring[df_coocurring["F90Cooccuring"] != "F90"]
        print(df_coocurring[['F90Cooccuring', 'F90CooccuringValue']].sort_values(by='F90CooccuringValue', ascending=False))


        # Make a set variable named data from df_coocurring dataframe that will have F90Cooccuring column as key and F90CooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)
        return self.calculate_and_draw_graph(G, data, F90_count)

    def calculate_and_draw_graph(self, G, data, F90_count):
            base_size = 300  # Base size for all nodes
            scale_factor = 50  # Scale factor for node sizes based on occurrences

            # Node sizes based on co-occurrence value
            node_sizes = {node: base_size + scale_factor * weight for node, weight in data.items()}
            node_sizes["F90"] = base_size + scale_factor * F90_count

            # Separate color for the central node "F90"
            central_node_color = "darkorange"

            # Exclude "F90" from the min and max calculations for normalization
            min_size = min(size for node, size in node_sizes.items() if node != "F90")
            max_size = max(size for node, size in node_sizes.items() if node != "F90")

            # Generate a continuous colormap for other nodes
            colormap = plt.cm.viridis
            normalize = mcolors.Normalize(vmin=min_size, vmax=max_size)
            scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)

            # Assign colors to nodes, excluding "F90" from affecting the color scale
            node_colors = {node: central_node_color if node == "F90" else scalar_map.to_rgba(size)
                        for node, size in node_sizes.items()}

            pos = nx.spring_layout(G)  # Position nodes using the spring layout

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[n] for n in G.nodes()], 
                                node_color=[node_colors[n] for n in G.nodes()], alpha=0.6)
            nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)  # Thicker edge lines

            # Drawing edge labels with custom font color and no background
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.3, font_size=10)

            nx.draw_networkx_labels(G, pos, font_size=12)
            plt.axis('off')
            return plt

########################################################
# 4-1- Level 1 - without color size adjustments
########################################################
class CoocurringDiagnosisNetworkGraph_higherlevel1:
    def load_plot_network_graph(self, original_df):
        # Load CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df")
        )
        # Select only diagnosis codes which are from 1 to 1306 , from 1306 onwards are ATC codes
        dummied_df = dummied_df.iloc[:, :1306]

        # Make a new dataframe containing only those rows, based on 'episode_id' in original_df
        processed_df = dummied_df[
            dummied_df["episode_id"].isin(original_df["episode_id"])
        ]

        # Merge dataframe with original_df by selecting pasient from original_df
        df = pd.merge(
            original_df[
                [
                    "pasient",
                    "episode_id",
                ]
            ],
            processed_df,
            on="episode_id",
        )

        # From the df dataframe find all the column names that starts with F90 and list them in a list as string
        F90 = [col for col in df.columns if col.startswith("F90")]

        # Create a column for F diagnosis
        df["F90"] = df[F90].max(axis=1)

        # Drop the original F columns
        df = df.drop(columns=F90)


         # Process diagnosis columns
        for diagnosis in df.columns[2:]:
            high_level_diagnosis = diagnosis[:1]  # Get the first 1 character

            # Identify all columns that belong to this high-level diagnosis
            related_columns = [col for col in df.columns if col.startswith(high_level_diagnosis)]
            
            # Recalculate the high-level diagnosis column by taking the max value across related detailed diagnoses
            df[high_level_diagnosis] = df[related_columns].max(axis=1)

        unique_high_level_diagnoses = set([col[:1] for col in df.columns[2:]])

        # Ensure df_coocurring is a DataFrame with the intended structure
        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])

        # Accumulate new rows in a list for efficient DataFrame construction
        new_rows = []

        for high_level_diagnosis in unique_high_level_diagnoses:
            # Count the unique patients with both the "F90" condition and the current high-level diagnosis
            count = df[(df["F90"] == 1) & (df[high_level_diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                new_row = {"F90Cooccuring": high_level_diagnosis, "F90CooccuringValue": count}
                new_rows.append(new_row)

        # Append all new rows at once to the df_coocurring DataFrame
        df_coocurring = pd.concat([df_coocurring, pd.DataFrame(new_rows)], ignore_index=True)

        # Check the count of F90CooccuringValue
        F90_count = int(df_coocurring[df_coocurring["F90Cooccuring"] == "F"]["F90CooccuringValue"].values)

        # Find the row in df_coocurring dataframe with the FCooccuring as F90 & exclude it from the df_coocurring dataframe
        df_coocurring = df_coocurring[df_coocurring["F90Cooccuring"] != "F"]


        # Make a set variable named data from df_coocurring dataframe that will have FCooccuring column as key and FCooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)
        return self.calculate_and_draw_graph(G, data, F90_count)

    def calculate_and_draw_graph(self, G, data, F90_count):
        # Sort nodes by weight in descending order
        sorted_nodes = sorted(data.items(), key=lambda x: x[1], reverse=True)

        # Calculate positions
        node_positions = {}
        angle_gap = 2 * np.pi / len(sorted_nodes)  # Gap between nodes
        for i, (node, weight) in enumerate(sorted_nodes):
            angle = i * angle_gap
            # Position nodes in a circle around (0, 0)
            # Adjust the radius or other parameters as needed
            x = np.cos(angle) * 10  # Radius of 10, adjust as needed
            y = np.sin(angle) * 10
            node_positions[node] = (x, y)

        # Ensure "F" is at the center
        node_positions["F90"] = (0, 0)

        # Draw the graph using calculated positions
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_edges(G, node_positions, edge_color="grey")
        nx.draw_networkx_labels(
            G,
            node_positions,
            font_size=8,
            labels={node: node for node in G.nodes() if node != "F90"},
        )

        # Draw the "F" node with a specific size
        nx.draw_networkx_nodes(
            G, node_positions, nodelist=["F90"], node_size=(F90_count * 50), node_color="orange"
        )

        # Draw other nodes with the default size
        nx.draw_networkx_nodes(
            G,
            node_positions,
            nodelist=[node for node in G.nodes() if node != "F90"],
            node_size= 500, 
            node_color="#60c3dd",
        )
        nx.draw_networkx_labels(G, node_positions, labels={"F90": "F90"}, font_size=20)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
        return plt

########################################################
# 4-2- Level 1 - adjusted
########################################################
class CoocurringDiagnosisNetworkGraph_higherlevel11:
    def load_plot_network_graph(self, original_df):
        # Load CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df")
        )
        # Select only diagnosis codes which are from 1 to 1306 , from 1306 onwards are ATC codes
        dummied_df = dummied_df.iloc[:, :1306]

        # Make a new dataframe containing only those rows, based on 'episode_id' in original_df
        processed_df = dummied_df[
            dummied_df["episode_id"].isin(original_df["episode_id"])
        ]

        # Merge dataframe with original_df by selecting pasient from original_df
        df = pd.merge(
            original_df[
                [
                    "pasient",
                    "episode_id",
                ]
            ],
            processed_df,
            on="episode_id",
        )

        # From the df dataframe find all the column names that starts with F90 and list them in a list as string
        F90 = [col for col in df.columns if col.startswith("F90")]

        # Create a column for F diagnosis
        df["F90"] = df[F90].max(axis=1)

        # Drop the original F columns
        df = df.drop(columns=F90)


         # Process diagnosis columns
        for diagnosis in df.columns[2:]:
            high_level_diagnosis = diagnosis[:1]  # Get the first 1 character

            # Identify all columns that belong to this high-level diagnosis
            related_columns = [col for col in df.columns if col.startswith(high_level_diagnosis)]
            
            # Recalculate the high-level diagnosis column by taking the max value across related detailed diagnoses
            df[high_level_diagnosis] = df[related_columns].max(axis=1)

        unique_high_level_diagnoses = set([col[:1] for col in df.columns[2:]])

        # Ensure df_coocurring is a DataFrame with the intended structure
        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])

        # Accumulate new rows in a list for efficient DataFrame construction
        new_rows = []

        for high_level_diagnosis in unique_high_level_diagnoses:
            # Count the unique patients with both the "F90" condition and the current high-level diagnosis
            count = df[(df["F90"] == 1) & (df[high_level_diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                new_row = {"F90Cooccuring": high_level_diagnosis, "F90CooccuringValue": count}
                new_rows.append(new_row)

        # Append all new rows at once to the df_coocurring DataFrame
        df_coocurring = pd.concat([df_coocurring, pd.DataFrame(new_rows)], ignore_index=True)

        # Check the count of F90CooccuringValue
        F90_count = int(df_coocurring[df_coocurring["F90Cooccuring"] == "F"]["F90CooccuringValue"].values)

        # Find the row in df_coocurring dataframe with the FCooccuring as F90 & exclude it from the df_coocurring dataframe
        df_coocurring = df_coocurring[df_coocurring["F90Cooccuring"] != "F"]


        # Make a set variable named data from df_coocurring dataframe that will have FCooccuring column as key and FCooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)
        return self.calculate_and_draw_graph(G, data, F90_count)

    def calculate_and_draw_graph(self, G, data, F90_count):
            base_size = 300  # Base size for all nodes
            scale_factor = 50  # Scale factor for node sizes based on occurrences

            # Node sizes based on co-occurrence value
            node_sizes = {node: base_size + scale_factor * weight for node, weight in data.items()}
            node_sizes["F90"] = base_size + scale_factor * F90_count

            # Separate color for the central node "F90"
            central_node_color = "darkorange"

            # Exclude "F90" from the min and max calculations for normalization
            min_size = min(size for node, size in node_sizes.items() if node != "F90")
            max_size = max(size for node, size in node_sizes.items() if node != "F90")

            # Generate a continuous colormap for other nodes
            colormap = plt.cm.viridis
            normalize = mcolors.Normalize(vmin=min_size, vmax=max_size)
            scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)

            # Assign colors to nodes, excluding "F90" from affecting the color scale
            node_colors = {node: central_node_color if node == "F90" else scalar_map.to_rgba(size)
                        for node, size in node_sizes.items()}

            pos = nx.spring_layout(G)  # Position nodes using the spring layout

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[n] for n in G.nodes()], 
                                node_color=[node_colors[n] for n in G.nodes()], alpha=0.6)
            nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)  # Thicker edge lines

            # Drawing edge labels with custom font color and no background
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.2, font_size=10)

            nx.draw_networkx_labels(G, pos, font_size=12)
            plt.axis('off')
            return plt
 