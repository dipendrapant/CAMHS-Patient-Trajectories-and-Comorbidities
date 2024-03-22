####################
# 1. Find all the columns that starts with F90 in the dataframe
# 2. List them in a list as string
# 3. Create a column for F90 diagnosis and count the number of patients that have both F90 and the other diagnosis
####################
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()

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
        self.calculate_and_draw_graph(G, data, F90_count)

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
        plt.figure(figsize=(40, 40))
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
        plt.show()


class CoocurringDiagnosisNetworkGraph_higherlevel3:
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
            # Create a temporary sum column for each high-level diagnosis count
            if high_level_diagnosis not in df:
                df[high_level_diagnosis] = df[[col for col in df.columns if col.startswith(high_level_diagnosis)]].max(axis=1)
            
        unique_high_level_diagnoses = set([col[:3] for col in df.columns[2:]])
       
        # Ensure df_coocurring is a DataFrame
        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])

        # Accumulate new rows in a list
        new_rows = []

        for high_level_diagnosis in unique_high_level_diagnoses:
            count = df[(df["F90"] == 1) & (df[high_level_diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                new_row = {"F90Cooccuring": high_level_diagnosis, "F90CooccuringValue": count}
                new_rows.append(new_row)

        # Append all new rows at once
        df_coocurring = pd.concat([df_coocurring, pd.DataFrame(new_rows)], ignore_index=True)
 
        # Check the count of F90CooccuringValue
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
        self.calculate_and_draw_graph(G, data, F90_count)

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
        plt.figure(figsize=(40, 40))
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
        plt.show()        

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


        # Step 1: Map each detailed diagnosis to a high-level category               
        # Process diagnosis columns
        for diagnosis in df.columns[2:]:
            high_level_diagnosis = diagnosis[:1]  # Get the first 3 characters
            # Create a temporary sum column for each high-level diagnosis count
            if high_level_diagnosis not in df:
                df[high_level_diagnosis] = df[[col for col in df.columns if col.startswith(high_level_diagnosis)]].max(axis=1)
            
        unique_high_level_diagnoses = set([col[:1] for col in df.columns[2:]])
        print(len(unique_high_level_diagnoses))

        # Ensure df_coocurring is a DataFrame
        df_coocurring = pd.DataFrame(columns=["F90Cooccuring", "F90CooccuringValue"])

        # Accumulate new rows in a list
        new_rows = []

        for high_level_diagnosis in unique_high_level_diagnoses:
            count = df[(df["F90"] == 1) & (df[high_level_diagnosis] == 1)]["pasient"].nunique()
            if count > 0:
                new_row = {"F90Cooccuring": high_level_diagnosis, "F90CooccuringValue": count}
                new_rows.append(new_row)

        # Append all new rows at once
        df_coocurring = pd.concat([df_coocurring, pd.DataFrame(new_rows)], ignore_index=True)
        print(df_coocurring['F90CooccuringValue'].sum())
        print((df_coocurring[df_coocurring['F90Cooccuring'].str.startswith('L')]))
        print(f"Lenght: {len(df_coocurring)}")
        print(df_coocurring.head(400))

        F90_count = int(
            df_coocurring[df_coocurring["F90Cooccuring"] == "F"][
                "F90CooccuringValue"
            ].values
        )

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
        self.calculate_and_draw_graph(G, data, F90_count)

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
        plt.figure(figsize=(40, 40))
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
            node_size=500,
            node_color="#60c3dd",
        )
        nx.draw_networkx_labels(G, node_positions, labels={"F90": "F90"}, font_size=20)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
        plt.show() 


##############################
# Network graph for medication
##############################
class CoocurringMedicationNetworkGraph:
    def load_plot_network_graph(self, original_df):
        # Load your CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(os.getenv("dummied_df"))
        # Drop column with index 1 to 1306 from dataframe dummied_df
        dummied_df.drop(dummied_df.columns[1:1307], axis=1, inplace=True)

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

        # Make a set variable named data from df_coocurring dataframe that will have F90Cooccuring	column as key and F90CooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges with weights as they are in the data
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)

        plt.figure(figsize=(30, 30))

        # Draw the graph
        pos = nx.spring_layout(G, weight="weight")
        # Draw all nodes and edges
        nx.draw_networkx_edges(G, pos, edge_color="grey")
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=8,
            labels={node: node for node in G.nodes() if node != "F90"},
        )

        # Draw the "F90" node with a specific size
        nx.draw_networkx_nodes(
            G, pos, nodelist=["F90"], node_size=(F90_count * 50), node_color="orange"
        )

        # Draw other nodes with the default size
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node for node in G.nodes() if node != "F90"],
            node_size=500,
            node_color="#60c3dd",
        )
        nx.draw_networkx_labels(G, pos, labels={"F90": "F90"}, font_size=20)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()


