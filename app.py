import streamlit as st

def main():
    st.title("Style Guide Assistant")
    st.write("Welcome to the Style Guide Assistant!")
    
    # File uploader
    pdf_files = st.file_uploader(
        "Upload your style guide PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    
    if pdf_files:
        st.write(f"Uploaded {len(pdf_files)} files")
        
    # Simple chat input
    user_input = st.text_input("Ask a question about the style guide:")
    if user_input:
        st.write(f"You asked: {user_input}")
        st.write("This is a placeholder response. Full functionality coming soon!")

if __name__ == "__main__":
    main()
