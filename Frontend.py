from Backend import mainGraph, interactions, showGraph, create_graph_from_post
import streamlit as st
from PIL import Image
from io import BytesIO

# Loading Image using PIL
im = Image.open('2.jpg')

# Adding Image to web app
st.set_page_config(page_title="Reddit Interactions Prediction App", page_icon = im)
# Streamlit app layout

logreg_node = mainGraph()

# Display the image in the top-left corner
st.markdown(
    """
    <style>
    .top-left {
        position: absolute;
        top: 10px;
        left: 10px;
    }
    .stApp {
        background: linear-gradient(135deg, #000000, #420296);
        margin: 0;
        padding: 0;
    }
    .title {
        color:black;
    } 
    </style>
    """,
    unsafe_allow_html=True
)

# Add a container for your app content
with st.container():

    # Add the image using Markdown
    st.markdown(
        '<div class="top-left"><img src="1.jpg"></div>',
        unsafe_allow_html=True
    )

    image = Image.open("1.jpg")

    # Display the image in the top-left corner
    st.image(image, use_column_width=False, width=100, caption='', clamp=False)

    st.markdown('<h1 style="color: black;">r/Romania Post Interactions</h1>', unsafe_allow_html=True)

    # Split the app into two columns
    col1, col2 = st.columns(2)

    # Add text input fields for URLs in each column
    with col1:
        post_url = st.text_input("", placeholder="Enter Post URL" )    

    # Perform actions based on the post URL
    if st.button("Process"):
        if post_url:
            graph = create_graph_from_post(post_url)
            graph = showGraph(graph)
            buf = BytesIO()
            graph.savefig(buf, format="png")
            st.image(buf, width=600)
            #st.pyplot(showGraph(graph))
            st.markdown('<h3 style="color: black; font-size: 20px;">Predicted number of interactions: {}</h3>'.format(interactions(post_url, logreg_node)), unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color: black; font-size: 20px;">Please enter a valid URL.</h2>', unsafe_allow_html=True)

st.markdown(
    """
<style>
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        max-width: 2000px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0px 0px 100px rgba(0, 0, 0, 0.8);
    }
</style>
""",
    unsafe_allow_html=True,
)