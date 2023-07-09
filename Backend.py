import praw
import matplotlib.pyplot as plt
import networkx as nx
from karateclub import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import streamlit as st

reddit = praw.Reddit(
    client_id='aaqo7P5fUXCJ1vvEHr_zXQ',
    client_secret='ATuvKXql-pzU6-7mqb-lqfuEs--J8w',
    user_agent='Robert, Andrada, IC project 1.0'
)

# Define the subreddit and create the graph
subreddit_name = 'Romania'

@st.cache_data()
def mainGraph():
    
    graph = nx.Graph()

    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.hot(limit=5):  # Adjust the limit as per your needs
        # Add submission as a node in the graph
        graph.add_node(submission.id, text=submission.title, type='submission')
        # Connect submission nodes with author nodes if author information is available
        if submission.author is not None:
            graph.add_edge(submission.id, submission.author.name, type='author')
        # Collect comments and connect them to their parent submissions or authors
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            graph.add_node(comment.id, text=comment.body, type='comment')
            # Connect comments with author nodes if author information is available
            if comment.author is not None:
                graph.add_edge(comment.id, comment.author.name, type='author')
            if comment.parent_id.startswith('t3_'):  # Parent is a submission
                graph.add_edge(comment.id, comment.parent_id, type='comment_to_submission')
            else:  # Parent is another comment
                graph.add_edge(comment.id, comment.parent_id[3:], type='comment_to_comment')

    #graph = load('graph_cache.joblib') # can use this for cached graphs

    # Create a new graph with relabeled nodes
    relabeled_graph = nx.convert_node_labels_to_integers(graph, first_label=0)

    # Create a Node2Vec model
    model = Node2Vec(dimensions=128)

    # Fit the model to the relabeled graph
    model.fit(relabeled_graph)

    # Get the computed node embeddings
    node_embeddings = model.get_embedding()

    # Manually assign labels based on node type
    node_labels = {}
    for node in graph.nodes:
        node_type = graph.nodes[node].get('type')
        if node_type == 'comment':
            node_labels[node] = 'interaction'
        else:
            node_labels[node] = 'non_interaction'

    # Convert node labels to numerical values
    label_encoder = LabelEncoder()
    encoded_node_labels = label_encoder.fit_transform(list(node_labels.values()))

    scaler = StandardScaler()
    node_embeddings_scaled = scaler.fit_transform(node_embeddings)

    # Split the data into training and testing sets
    X_train_node, X_test_node, y_train, y_test = train_test_split(node_embeddings_scaled, encoded_node_labels, test_size=0.2, random_state=42)

    # Create and train the logistic regression model for nodes
    logreg_node = LogisticRegression(max_iter=500)
    logreg_node.fit(X_train_node, y_train)

    # Calculate accuracy
    accuracy = logreg_node.score(X_test_node, y_test)

    # Calculate precision
    precision = precision_score(y_test, logreg_node.predict(X_test_node))

    # Calculate recall
    recall = recall_score(y_test, logreg_node.predict(X_test_node))

    # Calculate F1 score
    f1 = f1_score(y_test, logreg_node.predict(X_test_node))

    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return logreg_node

@st.cache_data()
# Create a function to create a graph from a post URL
def create_graph_from_post(url):
    graph = nx.Graph()
    submission = reddit.submission(url=url)
    # Add submission as a node in the graph
    graph.add_node(submission.id, text=submission.title, type='submission')
    # Connect submission nodes with author nodes if author information is available
    if submission.author is not None:
        graph.add_edge(submission.id, submission.author.name, type='author')
    # Collect comments and connect them to their parent submissions or authors
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        graph.add_node(comment.id, text=comment.body, type='comment')
        # Connect comments with author nodes if author information is available
        if comment.author is not None:
            graph.add_edge(comment.id, comment.author.name, type='author')
        if comment.parent_id.startswith('t3_'):  # Parent is a submission
            graph.add_edge(comment.id, comment.parent_id, type='comment_to_submission')
        else:  # Parent is another comment
            graph.add_edge(comment.id, comment.parent_id[3:], type='comment_to_comment')
    # Reindex the nodes in ascending order
    reindexed_graph = nx.convert_node_labels_to_integers(graph, ordering='sorted', label_attribute='old_label')

    return reindexed_graph

def showGraph(graph):
    pos = nx.kamada_kawai_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=100, node_color='blue')
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray', alpha=0.5)
    ax.axis('off')

    return fig

def create_embeddings(url):

    graph=create_graph_from_post(url)
    model=Node2Vec(dimensions=128)
    model.fit(graph)
    node_embeddings=model.get_embedding()

    return node_embeddings    

def interactions(url, logreg_node):
    # Gather Input Data
    #submission = reddit.submission(url)
    #text = submission.title + ' ' + submission.selftext

    label_encoder = LabelEncoder()
    label_encoder.fit(['non_interaction', 'interaction'])
    node_embeddings = create_embeddings(url)

    prediction_node = logreg_node.predict(node_embeddings)
    num_interactions_node = len(prediction_node[prediction_node == label_encoder.transform(['interaction'])])

    print("Number of interactions: ", num_interactions_node )

    return num_interactions_node