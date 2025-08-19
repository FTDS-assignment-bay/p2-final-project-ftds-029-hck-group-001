import streamlit as st


def show():
    image_khal = "https://raw.githubusercontent.com/Khalif-Coding/Picture-Repo/main/Khalif%20Santoso.png"
    image_damar = "https://raw.githubusercontent.com/Khalif-Coding/Picture-Repo/main/Nugroho%20Wicaksono.png"
    image_ian = "https://raw.githubusercontent.com/Khalif-Coding/Picture-Repo/main/Rd.%20Ladityarsa%20Ilyankusuma.png"

    # --- Custom CSS for the page layout ---
    st.markdown(
        """
        <style>
       
            
            /* Center the text and image within the column */
            .profile-card {
                text-align: center;
                /* Add a flex container to center the content vertically and horizontally */
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            /* Profile image styling */
            .profile-pic {
                border-radius: 50%; /* Makes the image circular */
                border: 2px solid #ccc;
                object-fit: cover;
                object-position: center 30%;
                width: 200px;
                height: 200px;
                margin: auto;
                display: block; /* Centers the image */
            }

            /* Profile name styling */
            .profile-name {
                margin-top: 15px;
                font-size: 1.2em;
                font-weight: 600;
                color: white;
            }

            .description-text {
                font-size: 1em;
                line-height: 1.6;
                color: #white;
                max-width: 800px;
                margin: 0 auto;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Application UI ---
    # Use a container to hold the entire profile section
    # with st.container():  <-- Removed this container
    st.markdown('<div class="profile-page-container">', unsafe_allow_html=True)
    
    # Create three columns for the profile pictures
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="profile-card">
                <img src= "{image_ian}" class="profile-pic">
                <p class="profile-name">Rd. Laditya Ilyankusuman</p>
                <p class="profile-name">Data Scientist</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="profile-card">
                <img src="{image_khal}" class="profile-pic">
                <p class="profile-name">Khalif Prabowo Santoso</p>
                <p class="profile-name">Data Analyst</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="profile-card">
                <img src="{image_damar}" class="profile-pic">
                <p class="profile-name">Nugroho Damar Wicaksono</p>
                <p class="profile-name">Data Engineer</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Add a horizontal line for separation
    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    # Use the custom HTML class for the description text
    st.markdown(
        """
        <div class="description-text">
            <p>E-Quinox is an interactive analytics platform built to explore customer behaviour patterns in e-commerce.
            Our team leverages real-world behavioural data — including product views, add-to-cart actions, and transactions — to gain actionable insights and support research in recommender systems using implicit feedback.
            This application brings together data science, visualization, and domain expertise to help understand customer journeys, identify trends, and drive smarter decision-making in digital retail.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Close the purple container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()
