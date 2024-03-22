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

load_dotenv()


class CoocurringDiagnosisNetworkGraph:
    def load_plot_network_graph(self, original_df):
        # Load your CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(
            "/mnt/work/workbench/dipendrp/new-data/Dummies_ICD10_ATC_All.csv"
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
