import streamlit as st

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #000000;
        padding-bottom: 20px;
    }
    .section-header {
        color: #000000;
        border-bottom: 2px solid #6200EE;
        padding-bottom: 10px;
        margin: 30px 10px;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6200EE;
        margin: 10px 0;
    }
    .highlight-text {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-title'>Face Authentication System</h1>", unsafe_allow_html=True)

# Brief Description
st.markdown("""
<div class="feature-box">
This advanced face recognition system provides a secure and modern approach to user authentication. 
By leveraging state-of-the-art facial recognition technology, it offers a seamless and contactless way to verify user identity.
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("<h2 class='section-header'>🌟 Key Features</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - 🎯 **Real-time Face Detection**
    - 🔒 **Secure Authentication**
    - 👥 **Multiple User Support**
    """)

with col2:
    st.markdown("""
    - ⚡ **Fast Processing Speed**
    - 💻 **User-friendly Interface**
    - 🔄 **Easy Registration Process**
    """)

# How to Use Section
st.markdown("<h2 class='section-header'>📖 How to Use</h2>", unsafe_allow_html=True)
st.markdown("""
1. **Registration**
   - Navigate to the registration page
   - Capture your facial data through webcam OR
   - Upload an image from files
   - Enter your name
   - Submit the registration

2. **Login**
   - Navigate to the login page
   - Position your face in the frame
   - Click 'Take Photo'
   - Click 'Login'
""")

# Use Cases Section
st.markdown("<h2 class='section-header'>📝 Use Cases</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🏢 Corporate
    - Office entry systems
    - Secure workspace access
    - Employee attendance
    """)

with col2:
    st.markdown("""
    ### 🏥 Healthcare
    - Patient identification
    - Staff authentication
    - Restricted area access
    """)

with col3:
    st.markdown("""
    ### 🏫 Educational
    - Student attendance
    - Exam verification
    - Library access
    """)

# Security Notes
st.markdown("<h2 class='section-header'>🔐 Security Features</h2>", unsafe_allow_html=True)
st.markdown("""
<div class="feature-box">
- Advanced encryption for facial data<br>
- Secure database management<br>
- Privacy-focused design
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️</p>
    <p>For support or queries, please contact the system <a href="https://www.linkedin.com/in/arslaan365/">administrator</a></p>
</div>
""", unsafe_allow_html=True)
