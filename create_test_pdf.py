#!/usr/bin/env python3

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
import sys

def create_test_pdf():
    """Create a simple test PDF for testing purposes"""
    
    # Sample content about machine learning
    content = [
        "Machine Learning and Artificial Intelligence",
        "",
        "This is a test ebook about machine learning and artificial intelligence.",
        "It covers various important topics in the field of AI and ML.",
        "",
        "Chapter 1: Neural Networks",
        "Neural networks are the foundation of deep learning. They consist of",
        "interconnected nodes that process information similar to brain neurons.",
        "",
        "Chapter 2: Deep Learning Algorithms", 
        "Deep learning uses multi-layered neural networks to learn complex",
        "patterns in data. Popular algorithms include CNN, RNN, and transformers.",
        "",
        "Chapter 3: Data Science",
        "Data science combines statistics, programming, and domain expertise",
        "to extract insights from large datasets.",
        "",
        "Chapter 4: Computer Vision",
        "Computer vision enables machines to interpret and understand visual",
        "information from the world around them.",
        "",
        "Keywords: machine learning, artificial intelligence, neural networks,",
        "deep learning, data science, computer vision, algorithms, supervised",
        "learning, unsupervised learning, reinforcement learning, natural",
        "language processing, CNN, RNN, transformers"
    ]
    
    # Create PDF
    pdf_path = "/home/joerg/code/lit_DB/test_literature/sample_ml_book.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Set font
    c.setFont("Helvetica", 12)
    
    # Add content
    y_position = 750
    for line in content:
        if y_position < 50:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750
            
        c.drawString(50, y_position, line)
        y_position -= 20
    
    c.save()
    print(f"✅ Created test PDF: {pdf_path}")

if __name__ == "__main__":
    try:
        create_test_pdf()
    except ImportError:
        print("❌ reportlab not installed. Installing...")
        os.system("pip install reportlab")
        create_test_pdf()