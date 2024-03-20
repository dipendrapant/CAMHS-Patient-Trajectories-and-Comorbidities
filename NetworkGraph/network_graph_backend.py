####################
# 1. Find all the columns that starts with F90 in the dataframe
# 2. List them in a list as string
# 3. Create a column for F90 diagnosis and count the number of patients that have both F90 and the other diagnosis
####################
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class CoocurringNetworkGraph:
    def load_plot_network_graph(self, original_df):
        # Load your CSV data
        original_df = original_df
        print(original_df.shape)
        dummied_df = pd.read_csv(
            "/mnt/work/workbench/dipendrp/new-data/Dummies_ICD10_ATC4_All.csv"
        )

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

        # drop the last row of df_coocurring dataframe
        df_coocurring = df_coocurring[:-1]

        # Make a set variable named data from df_coocurring dataframe that will have F90Cooccuring	column as key and F90CooccuringValue as corresponding value
        data = dict(
            zip(df_coocurring["F90Cooccuring"], df_coocurring["F90CooccuringValue"])
        )
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges with weights
        for node, weight in data.items():
            G.add_edge("F90", node, weight=weight)

        # Define node colors
        node_colors = ["red" if node == "F90" else "skyblue" for node in G.nodes()]

        # Define the size of the plot
        plt.figure(figsize=(30, 20))

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            font_size=7,
            node_size=500,
            edge_color="grey",
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
