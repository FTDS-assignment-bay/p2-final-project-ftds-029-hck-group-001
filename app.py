import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import eda
import clustering
import bio

# Set up the page configuration
st.set_page_config(layout="wide", page_title="E-quinox Deployment")

# Initialize session state to control which screen is shown
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True

# --- CSS for styling the welcome screen ---
st.markdown(
    """
    <style>
    /* Use a dark theme for the entire page */
    .stApp {
        background-color: #0c0e12;
        color: #e0e0e0;
    }
    
   .centered-container{
        position: fixed;         /* detach from Streamlit's left column */
        inset: 50% auto auto 50%;
        transform: translate(-50%, -50%); /* perfect middle */
        width: min(900px, 90vw);
        text-align: center;

        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 16px;

        
    
    /* Style for the logo container */
    .logo-container {
        background-color: #1a1e26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        margin-bottom: 30px;
    }
    
    h1 {
        font-size: 2.5rem;
        color: #f0f2f6;
        margin-bottom: 5px;
    }
    
    p {
        font-size: 1rem;
        color: #a0a0a0;
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background-color: #white;
        color: white;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Logic to display either the welcome page or the main app ---

if st.session_state.show_welcome:
    # This block displays the welcome screen
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    
    # Load and display the logo
   # The URL for your logo
    logo_url = 'https://raw.githubusercontent.com/Khalif-Coding/Picture-Repo/main/Gemini_Generated_Image_padbhvpadbhvpadb.png'

    # Display the logo directly from the URL
    st.image(logo_url, width=200)

    st.markdown('<h1 class="welcome-text">Welcome to E-quinox</h1>', unsafe_allow_html=True)
    st.markdown("Your personalized shopping experience starts here.")
    st.write("") 

    # When the button is clicked, set the state to False to switch to the main app
    if st.button("🚀 Start"):
        st.session_state.show_welcome = False
        st.rerun() # Use st.rerun() to immediately re-run the script
        
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # This block displays the main application with the option menu
    st.title("E-Quinox Deployment")
    option = option_menu(None,['Profile',"Exploratory Data Analysis", "Tasks"], 
                        icons=['person', "bar-chart", 'bi-clipboard-check'], 
                        menu_icon="cast", 
                        default_index=0, 
                        orientation="horizontal",
                        styles = {"container": {"padding": "0!important"}})

    if option == "Profile":
        bio.show()                     
    elif option == "Exploratory Data Analysis":
        eda.show()
    elif option == "Tasks":
        clustering.show()
   