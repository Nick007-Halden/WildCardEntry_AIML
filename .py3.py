import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

def generate_description(keyword):
    """Generate descriptions based on provided keywords."""
    descriptions = {
        "leadership": "Demonstrated exceptional leadership skills by managing diverse teams and driving projects to success.",
        "data analysis": "Proficient in analyzing complex datasets to derive actionable insights and support decision-making.",
        "problem-solving": "Skilled in identifying challenges and implementing innovative solutions to achieve objectives.",
        "communication": "Exceptional verbal and written communication skills, ensuring effective collaboration across teams.",
        # Add more keyword mappings here...
    }
    return descriptions.get(keyword.lower(), f"Description not available for '{keyword}'.")

def create_pdf(data, file_name="resume_with_photo.pdf"):
    """Generate a PDF resume with keyword descriptions and photo."""
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Header Section
    header = Paragraph(f"<b>{data['name']}</b>", styles['Title'])
    contact_info = Paragraph(f"""
        <br/><b>Contact</b>: {data['phone']} | {data['email']}<br/>
        <b>LinkedIn</b>: {data['linkedin']}<br/>
        <b>Location</b>: {data['location']}
    """, styles['Normal'])
    elements.extend([header, contact_info])

    # Add Photo
    if data['photo']:
        elements.append(Image(data['photo'], width=100, height=100))

    elements.append(Paragraph("<br/><b>CAREER OBJECTIVE</b>", styles['Heading2']))
    elements.append(Paragraph(data['career_objective'], styles['Normal']))

    # Education Section
    elements.append(Paragraph("<br/><b>EDUCATION</b>", styles['Heading2']))
    for edu in data['education']:
        elements.append(Paragraph(f"<b>{edu['degree']}</b> - {edu['institution']} ({edu['year']})", styles['Normal']))

    # Core Competencies Section
    elements.append(Paragraph("<br/><b>CORE COMPETENCIES</b>", styles['Heading2']))
    for comp in data['core_competencies']:
        elements.append(Paragraph(f"- {generate_description(comp)}", styles['Normal']))

    # Internships Section
    elements.append(Paragraph("<br/><b>INTERNSHIPS</b>", styles['Heading2']))
    for internship in data['internships']:
        elements.append(Paragraph(f"<b>{internship['role']}</b> at {internship['organization']} ({internship['duration']})<br/>{internship['description']}", styles['Normal']))

    # Skills Section
    elements.append(Paragraph("<br/><b>SKILLS</b>", styles['Heading2']))
    elements.append(Paragraph(f"<b>Hard Skills</b>: {', '.join(data['hard_skills'])}", styles['Normal']))
    elements.append(Paragraph(f"<b>Soft Skills</b>: {', '.join(data['soft_skills'])}", styles['Normal']))

    # Achievements Section
    elements.append(Paragraph("<br/><b>ACHIEVEMENTS</b>", styles['Heading2']))
    for achievement in data['achievements']:
        elements.append(Paragraph(f"- {achievement}", styles['Normal']))

    # Certifications Section
    elements.append(Paragraph("<br/><b>CERTIFICATIONS</b>", styles['Heading2']))
    for cert in data['certifications']:
        elements.append(Paragraph(f"- {cert}", styles['Normal']))

    # Build PDF
    doc.build(elements)

def main():
    st.title("Advanced Resume Generator with Keywords and Photo")
    st.write("Fill in the details below to generate your resume with a photo and keyword descriptions.")

    # Input Fields
    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    linkedin = st.text_input("LinkedIn Profile URL")
    location = st.text_input("Location")

    # Photo Upload
    photo = st.file_uploader("Upload a photo for your resume", type=["jpg", "png"])

    st.subheader("Career Objective")
    career_objective = st.text_area("Write your career objective")

    st.subheader("Education")
    num_education = st.number_input("Number of Education Entries", min_value=1, max_value=5, step=1, value=1)
    education = []
    for i in range(int(num_education)):
        st.write(f"Education Entry {i+1}")
        degree = st.text_input(f"Degree (Education {i+1})", key=f"degree_{i}")
        institution = st.text_input(f"Institution (Education {i+1})", key=f"institution_{i}")
        year = st.text_input(f"Year (Education {i+1})", key=f"year_{i}")
        education.append({"degree": degree, "institution": institution, "year": year})

    st.subheader("Core Competencies")
    core_competencies = st.text_area("List core competencies (comma-separated)").split(",")

    st.subheader("Internships")
    num_internships = st.number_input("Number of Internships", min_value=0, max_value=5, step=1, value=1)
    internships = []
    for i in range(int(num_internships)):
        st.write(f"Internship {i+1}")
        role = st.text_input(f"Role (Internship {i+1})", key=f"role_{i}")
        organization = st.text_input(f"Organization (Internship {i+1})", key=f"organization_{i}")
        duration = st.text_input(f"Duration (Internship {i+1})", key=f"duration_{i}")
        description = st.text_area(f"Description (Internship {i+1})", key=f"description_{i}")
        internships.append({"role": role, "organization": organization, "duration": duration, "description": description})

    st.subheader("Skills")
    hard_skills = st.text_input("List hard skills (comma-separated)").split(",")
    soft_skills = st.text_input("List soft skills (comma-separated)").split(",")

    st.subheader("Achievements")
    achievements = st.text_area("List achievements (one per line)").split("\n")

    st.subheader("Certifications")
    certifications = st.text_area("List certifications (one per line)").split("\n")

    # Generate Resume Button
    if st.button("Generate Resume"):
        if name and phone and email and linkedin and location and career_objective:
            data = {
                "name": name,
                "phone": phone,
                "email": email,
                "linkedin": linkedin,
                "location": location,
                "career_objective": career_objective,
                "education": education,
                "core_competencies": [comp.strip() for comp in core_competencies if comp.strip()],
                "internships": internships,
                "hard_skills": [skill.strip() for skill in hard_skills if skill.strip()],
                "soft_skills": [skill.strip() for skill in soft_skills if skill.strip()],
                "achievements": [ach.strip() for ach in achievements if ach.strip()],
                "certifications": [cert.strip() for cert in certifications if cert.strip()],
                "photo": photo
            }
            create_pdf(data, "resume_with_photo.pdf")
            st.success("Resume generated successfully!")
            with open("resume_with_photo.pdf", "rb") as pdf_file:
                st.download_button(
                    label="Download Resume",
                    data=pdf_file,
                    file_name="resume_with_photo.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Please fill in all required fields.")

if __name__ == "__main__":
    main()
