#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from nltk import pos_tag
import time  # For spinners
import plotly.express as px  # For visualizations

# NLTK setup
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')


# Load Models and Preprocessing Tools
model_files = {
    "Logistic Regression": "app/Logistic_Regression.pkl",
    "SVM (Linear)": "app/SVM_Linear.pkl",
    "SVM (RBF)": "app/SVM_RBF.pkl",
    "Naive Bayes": "app/Naive_Bayes.pkl",
    "Random Forest": "app/Random_Forest.pkl",
    "Decision Tree": "app/Decision_Tree.pkl"
}

models = {}
for name, filename in model_files.items():
    with open(filename, "rb") as file:
        models[name] = pickle.load(file)

# Load TF-IDF model
tfidf = pickle.load(open('app/tfidf.pkl', 'rb'))


# List of predefined skills
skills =  [
        "Python", "Django", "Flask", "FastAPI", "Object-Oriented Programming (OOP)",
        "SQL", "NoSQL (e.g., MongoDB)", "Data Structures & Algorithms",
        "RESTful APIs", "GraphQL", "Unit Testing", "Pytest",
        "Git", "Version Control", "Cloud Services (AWS, Azure, GCP)",
        "Docker", "Kubernetes", "Data Analysis", "Pandas", "NumPy"
,
        "Java", "Spring", "Hibernate", "Java Servlets", "JSP",
        "RESTful APIs", "Microservices", "SQL (MySQL, PostgreSQL)",
        "Git", "Maven", "Jenkins", "Object-Oriented Programming (OOP)",
        "Testing (JUnit, TestNG)", "CI/CD", "Docker", "Kubernetes",
        "Design Patterns", "Multithreading", "Concurrency"
,
        "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js",
        "TypeScript", "SASS/SCSS", "Responsive Web Design", "Bootstrap",
        "Webpack", "Babel", "Git", "Version Control",
        "UI/UX Design Principles", "RESTful APIs", "GraphQL",
        "Testing (Jest, Mocha)", "Accessibility", "Cross-Browser Compatibility"
,
        "TCP/IP", "DNS", "DHCP", "Network Security", "Firewalls",
        "Cisco Routers & Switches", "VPN", "VLAN", "VoIP",
        "Network Monitoring (Wireshark, Nagios)", "IP Addressing", "Subnetting",
        "Troubleshooting", "Network Protocols (HTTP, FTP, SNMP)",
        "Linux/Unix Administration", "Backup & Disaster Recovery"
,
        "Project Planning", "Scheduling", "Agile", "Scrum", "Waterfall Methodologies",
        "Budgeting", "Resource Allocation", "Risk Management", "Team Leadership",
        "Communication", "Stakeholder Management", "JIRA", "Microsoft Project", "Trello",
        "Problem-Solving", "Reporting", "Documentation", "Time Management"
,
        "Firewalls", "VPN", "IDS/IPS", "Encryption", "PKI", "SSL/TLS",
        "SIEM Tools (Splunk, IBM QRadar)", "Threat Analysis", "Vulnerability Assessment",
        "Incident Response", "Forensics", "Penetration Testing (Nmap, Metasploit)",
        "Ethical Hacking", "OSI Model", "Cloud Security (AWS, Azure)",
        "Compliance (ISO, NIST, GDPR)", "Scripting (Python, Bash)"
,
        "Programming Languages (Python, Java, C++)", "Data Structures", "Algorithms",
        "Object-Oriented Programming (OOP)", "Version Control (Git)", "Software Testing",
        "Agile Methodologies", "Scrum", "Unit Testing", "CI/CD", "Database Management",
        "Cloud Computing (AWS, Azure, GCP)", "Docker", "Kubernetes"
,
        "Linux", "Windows Server", "Active Directory", "DNS", "DHCP", "Network Troubleshooting",
        "Server Maintenance", "Virtualization (VMware, Hyper-V)", "Scripting (Bash, PowerShell)",
        "Backup & Recovery", "Firewall Configuration", "Load Balancing"
,
        "HTML", "CSS", "JavaScript", "Responsive Design", "UI/UX Principles",
        "Version Control (Git)", "REST APIs", "GraphQL", "Frontend Frameworks (React, Vue, Angular)",
        "Backend Development (Node.js, Django)", "Database (SQL, NoSQL)", "Cross-Browser Compatibility"
,
        "SQL", "Database Design", "Database Optimization", "Backup & Recovery",
        "MySQL", "PostgreSQL", "Oracle Database", "NoSQL (MongoDB)", "Data Security",
        "Performance Tuning", "Stored Procedures", "Data Warehousing"
,
        "Python", "R", "Data Analysis", "Machine Learning", "Statistics",
        "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "Keras", "Data Visualization",
        "Matplotlib", "Seaborn", "SQL", "Big Data (Hadoop, Spark)", "Deep Learning"
,
        "Talent Acquisition", "Employee Relations", "Performance Management",
        "Recruitment Strategies", "Training & Development", "HR Policies",
        "Payroll Management", "Employee Engagement", "Conflict Resolution",
        "HR Software (Workday, SAP SuccessFactors)", "Labor Laws", "Compensation & Benefits",
        "Onboarding & Offboarding", "Employee Satisfaction", "HR Analytics"
,
        "Legal Research", "Litigation", "Case Management", "Negotiation",
        "Contract Drafting", "Legal Writing", "Client Communication", "Court Representation",
        "Criminal Law", "Civil Law", "Corporate Law", "Legal Compliance", "Intellectual Property",
        "Dispute Resolution", "Regulatory Compliance", "Mediation"
,
        "Creative Design", "Graphic Design", "Illustration", "Painting", "Sculpture",
        "Digital Art", "Animation", "Photography", "Art Direction", "Art History",
        "Visual Storytelling", "Typography", "Adobe Creative Suite", "Concept Art",
        "Printmaking", "Art Curation"
 ,
        "CAD Software (AutoCAD, SolidWorks)", "Finite Element Analysis (FEA)", "Thermodynamics",
        "Fluid Mechanics", "Heat Transfer", "Material Science", "Manufacturing Processes",
        "Mechanical Design", "Product Development", "Engineering Drawing", "3D Modeling",
        "Automation", "Mechatronics", "Robotics", "Vibration Analysis", "Project Management"
,
        "Sales Strategy", "Lead Generation", "Sales Presentations", "Negotiation Skills",
        "Customer Relationship Management (CRM)", "Market Research", "B2B Sales", "B2C Sales",
        "Account Management", "Sales Forecasting", "Closing Deals", "Cold Calling",
        "Sales Pipeline Management", "Product Knowledge", "Client Negotiation"
,
        "Personal Training", "Nutrition", "Exercise Physiology", "Fitness Assessment",
        "Strength Training", "Cardio Training", "Group Fitness Classes", "Health Coaching",
        "Wellness Programs", "Weight Loss", "Sports Science", "Yoga", "Physical Therapy",
        "Motivational Skills", "Injury Prevention", "Certified Personal Trainer (CPT)"
,
        "Structural Analysis", "Geotechnical Engineering", "Construction Materials",
        "Soil Mechanics", "Transportation Engineering", "Hydraulics", "Building Codes",
        "Project Management", "Surveying", "AutoCAD", "Revit", "Concrete and Steel Design",
        "Soil Testing", "Environmental Engineering", "Sustainable Design"
,
        "Business Requirements Gathering", "Data Analysis", "Business Process Modeling",
        "SQL", "Requirements Documentation", "Business Intelligence", "Stakeholder Management",
        "Agile Methodologies", "JIRA", "Data Visualization (Power BI, Tableau)",
        "SWOT Analysis", "Market Research", "Financial Analysis", "Risk Analysis",
        "Business Solutions Design", "Change Management"
,
        "SAP ABAP", "SAP Fiori", "SAP Hana", "SAP UI5", "SAP S/4HANA", "SAP PI/PO",
        "SAP CRM", "SAP BW", "SAP MM", "SAP SD", "SAP BASIS", "SAP Integration",
        "SQL", "Data Migration", "SAP Cloud Platform", "SAP Development Tools"
,
        "Test Automation Frameworks (Selenium, Appium, Cypress)", "Continuous Integration/Continuous Deployment (CI/CD)",
        "Test Scripts", "Software Testing", "Quality Assurance", "Test Management Tools (JIRA, TestRail)",
        "Python", "Java", "Automated Test Design", "Load Testing", "Regression Testing",
        "Test Execution", "Bug Reporting", "Version Control (Git)", "Performance Testing"
,
        "Circuit Design", "Power Systems", "Control Systems", "Microcontrollers", "Electronics",
        "Signal Processing", "Power Electronics", "Matlab", "Simulink", "PCB Design",
        "Electrical Safety", "Embedded Systems", "VLSI Design", "Renewable Energy Systems",
        "Electromagnetic Field Theory", "Digital Logic"
,
        "Process Optimization", "Supply Chain Management", "Project Management",
        "Resource Allocation", "Team Management", "Inventory Management", "Budgeting",
        "Operational Efficiency", "Risk Management", "Quality Control", "Logistics",
        "Vendor Management", "Cross-functional Collaboration", "Data Analysis", "Lean Manufacturing"
,
        "CI/CD", "Automation", "Linux/Unix Systems", "Cloud Platforms (AWS, Azure, GCP)",
        "Docker", "Kubernetes", "Terraform", "Infrastructure as Code", "Monitoring & Logging (Prometheus, Grafana)",
        "Version Control (Git)", "Jenkins", "Ansible", "Python", "Scripting", "Microservices"
,
        "Project Planning", "Portfolio Management", "Risk Management", "Project Governance",
        "Project Scheduling", "Resource Management", "Budgeting", "Project Reporting",
        "Agile & Waterfall Methodologies", "Stakeholder Management", "Microsoft Project",
        "JIRA", "Risk Assessment", "Change Management", "Quality Assurance", "Project Documentation"
,
        "SQL", "Database Management", "Database Design", "Normalization", "Indexing",
        "Query Optimization", "Data Warehousing", "ETL", "Database Administration", 
        "Data Backup & Recovery", "Replication", "NoSQL (MongoDB, Cassandra)", 
        "Cloud Databases (AWS RDS, Azure SQL)", "Data Integrity", "Database Security"
,
        "Hadoop Ecosystem (HDFS, YARN, MapReduce)", "Apache Hive", "Apache Pig", "HBase", 
        "Apache Spark", "Hadoop Distributed File System (HDFS)", "Data Lakes", "Batch Processing",
        "Real-Time Data Processing", "Apache Kafka", "Cloudera", "Big Data Technologies",
        "Data Engineering", "ETL Pipelines", "Data Integration"
,
        "ETL Tools (Informatica, Talend, Apache Nifi)", "Data Integration", "SQL", "Data Warehousing",
        "Data Transformation", "Data Cleansing", "Data Migration", "Data Loading", "Data Pipelines",
        "Apache Kafka", "Spark", "SQL Server Integration Services (SSIS)", "Process Automation",
        "Business Intelligence", "Big Data Technologies", "Python for ETL"
,
        "C#", ".NET Framework", ".NET Core", "ASP.NET", "Web APIs", "MVC Architecture", "SQL Server",
        "Entity Framework", "LINQ", "Azure", "Web Development", "JavaScript", "HTML/CSS", 
        "Agile Methodologies", "Unit Testing (NUnit, MSTest)", "Microservices"
,
        "Blockchain Architecture", "Smart Contracts", "Ethereum", "Bitcoin", "Cryptocurrency",
        "Distributed Ledger Technology", "Solidity", "Blockchain Security", "Decentralized Applications (dApps)",
        "Consensus Algorithms", "Hyperledger", "Smart Contract Development", "Blockchain APIs", 
        "IPFS", "Cryptographic Algorithms", "Decentralized Finance (DeFi)"
,
        "Manual Testing", "Automation Testing", "Selenium", "TestNG", "JIRA", "Bug Tracking", 
        "Performance Testing", "Load Testing", "Regression Testing", "Unit Testing", "API Testing",
        "Postman", "Quality Assurance", "Continuous Integration", "Test Automation Frameworks"
,
        "Server-Side Development", "APIs (RESTful, SOAP)", "Database Management", "Node.js", "Python",
        "Java", "Ruby", "PHP", "SQL", "NoSQL Databases", "Authentication", "Cloud Technologies",
        "Microservices", "Version Control (Git)", "Docker", "CI/CD", "Security Best Practices"
,
        "Frontend Technologies (HTML, CSS, JavaScript)", "Backend Technologies (Node.js, Python, Ruby)",
        "Databases (SQL, NoSQL)", "RESTful APIs", "Version Control (Git)", "Agile Methodologies", 
        "Cloud Platforms (AWS, Azure, GCP)", "React, Angular, Vue.js", "Web Development", "CSS Preprocessors",
        "Microservices", "Testing (Jest, Mocha)", "Continuous Integration", "Deployment Pipelines", "Docker"
,
        "Swift (iOS)", "Kotlin (Android)", "Objective-C", "Android Studio", "Xcode", "Mobile UI/UX Design",
        "Mobile Development Frameworks (React Native, Flutter)", "REST APIs", "Version Control (Git)",
        "Firebase", "SQL", "NoSQL Databases", "Push Notifications", "App Store Submission", "Cross-Platform Development",
        "Unit Testing (JUnit, XCTest)", "UI Testing"
,
        "Machine Learning Algorithms (Linear Regression, Decision Trees, SVM)", "Deep Learning", 
        "Neural Networks", "NLP", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", 
        "NumPy", "Data Preprocessing", "Feature Engineering", "Model Deployment", "Model Evaluation",
        "Cloud Platforms (AWS SageMaker, Azure ML)", "Big Data Technologies", "Reinforcement Learning"
,
        "Cloud Platforms (AWS, Azure, GCP)", "Cloud Architecture", "Infrastructure as Code (Terraform)",
        "Containers (Docker, Kubernetes)", "Cloud Security", "CI/CD", "Virtualization", "Serverless Computing",
        "DevOps", "Automation", "Cloud Networking", "Cost Optimization", "Data Storage (S3, Blob Storage)",
        "Disaster Recovery", "Cloud Monitoring & Management"
,
        "Graphic Design", "UI/UX Design", "Web Design", "Wireframing", "Prototyping", "Adobe Creative Suite",
        "Illustrator", "Photoshop", "Sketch", "Figma", "InVision", "Responsive Design", "Typography",
        "Creative Direction", "Branding", "User Research"
,
        "IT Support", "Troubleshooting", "Networking", "Cloud Computing", "Security", "Database Management",
        "Systems Administration", "IT Infrastructure", "Disaster Recovery", "Help Desk", "Virtualization",
        "Firewall Management", "Software Development", "Project Management", "Business Continuity Planning",
        "Network Administration"
,
        "Classroom Management", "Lesson Planning", "Curriculum Development", "Educational Technology", 
        "Assessment & Evaluation", "Student Engagement", "Communication Skills", "Differentiated Instruction", 
        "Subject Matter Expertise", "Time Management", "Mentoring", "Instructional Design", "Classroom Organization", 
        "Parent-Teacher Communication", "Pedagogical Knowledge"
,
        "Sales Strategy", "Market Research", "Lead Generation", "Negotiation", "Salesforce", 
        "Client Relationship Management", "Sales Forecasting", "Customer Acquisition", "Networking", 
        "Revenue Growth", "Partnerships & Alliances", "Business Strategy", "Project Management", "Contract Management", 
        "Presentation Skills"
,
        "Medical Knowledge", "Patient Care", "Clinical Skills", "Health Informatics", "Medical Coding & Billing", 
        "Healthcare Administration", "Patient Advocacy", "HIPAA Compliance", "Electronic Health Records (EHR)", 
        "Nursing Skills", "Medical Research", "Pharmacology", "Laboratory Skills", "Healthcare IT", "Healthcare Regulations"
,
        "Crop Production", "Soil Management", "Agricultural Machinery", "Irrigation Systems", "Pest Management", 
        "Agrochemical Knowledge", "Farm Management", "Agricultural Engineering", "Sustainability Practices", 
        "Livestock Management", "Agroforestry", "Supply Chain Management", "Agricultural Economics", 
        "Precision Agriculture", "Climate Change Adaptation"
,
        "Customer Service", "Client Relationship Management", "Call Center Operations", "Problem-Solving", 
        "Multitasking", "Communication Skills", "Data Entry", "Telemarketing", "CRM Software", "Team Management", 
        "Outsourcing", "Quality Assurance", "Inbound/Outbound Calling", "Training and Development", "Process Improvement"
,
        "Market Analysis", "Business Strategy", "Problem-Solving", "Stakeholder Management", "Process Improvement", 
        "Change Management", "Project Management", "Client Relationship", "Data Analysis", "Risk Management", 
        "Research Skills", "Financial Modeling", "Contract Negotiation", "Business Development", "Report Writing"
,
        "Content Creation", "Social Media Marketing", "SEO/SEM", "Video Production", "Graphic Design", "Brand Strategy", 
        "Content Marketing", "Web Analytics", "PPC Campaigns", "Influencer Marketing", "Email Marketing", 
        "Audience Engagement", "Storytelling", "Digital Advertising", "Data Analysis"
,
        "Mechanical Engineering", "Vehicle Maintenance", "Automotive Design", "Diagnostics", "AutoCAD", 
        "Automobile Manufacturing", "Quality Control", "Vehicle Systems", "Fuel Efficiency", "Sustainability", 
        "Electric Vehicles", "Automotive Repair", "Supply Chain Management", "Consumer Behavior", "Product Development"
,
        "Culinary Skills", "Food Preparation", "Menu Planning", "Food Safety & Sanitation", "Pastry & Baking", 
        "Recipe Development", "Cost Control", "Kitchen Management", "Nutritional Knowledge", "Catering", "Food Presentation", 
        "Teamwork", "Customer Service", "Inventory Management", "Wine Pairing"
,
        "Financial Analysis", "Accounting", "Investment Management", "Risk Management", "Financial Modeling", 
        "Budgeting", "Cash Flow Management", "Tax Planning", "Auditing", "Corporate Finance", "Mergers & Acquisitions", 
        "Regulatory Compliance", "Portfolio Management", "Business Valuation", "Financial Reporting"
,
        "Fashion Design", "Textile Knowledge", "Apparel Manufacturing", "Product Development", "Trends Analysis", 
        "Pattern Making", "Sourcing", "Fabrication", "Branding", "Visual Merchandising", "Fashion Illustration", 
        "Retail Management", "Supply Chain Management", "Costing", "Sustainability Practices"
,
        "Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Structural Analysis", "Design & Drafting", 
        "AutoCAD", "Project Management", "Thermodynamics", "Systems Engineering", "Manufacturing Processes", "Quality Assurance", 
        "Robotics", "Materials Science", "Data Analysis", "Simulation & Modeling"
,
        "Financial Reporting", "Bookkeeping", "Taxation", "Budgeting", "General Ledger Management", "Auditing", 
        "Accounts Payable & Receivable", "Payroll", "Financial Analysis", "Compliance", "Cost Accounting", 
        "Excel (Advanced)", "Accounting Software (QuickBooks, SAP)", "Forecasting", "Risk Management"
,
        "Project Management", "Construction Management", "Building Codes", "Cost Estimation", "Contract Negotiation", 
        "Blueprint Reading", "Scheduling", "Safety Regulations", "Site Supervision", "Materials Procurement", 
        "Civil Engineering", "Quality Control", "Sustainability Practices", "Structural Design", "Team Leadership"
   ,
        "Media Relations", "Content Creation", "Event Management", "Crisis Communication", "Social Media Management", 
        "Brand Messaging", "Press Releases", "Marketing Strategy", "Stakeholder Engagement", "Public Speaking", 
        "Storytelling", "Research Skills", "Networking", "Reputation Management", "Communication Strategy"
  ,
        "Risk Management", "Loan Processing", "Financial Products", "Compliance", "Investment Strategies", 
        "Wealth Management", "Customer Service", "Treasury Management", "Credit Analysis", "Foreign Exchange", 
        "Regulatory Knowledge", "Asset Management", "Financial Planning", "Branch Management", "Retail Banking"
    ,
        "Aviation Safety", "Aircraft Maintenance", "Pilot Training", "Air Traffic Control", "Aviation Regulations", 
        "Flight Planning", "Navigation Systems", "Aerospace Engineering", "Flight Operations", "Airline Management", 
        "Customer Service", "Crew Coordination", "Aviation Security", "Aircraft Systems", "Aviation Logistics"
    ]


# Map prediction to category name
category_mapping = {
    47: 'Python Developer', 37: 'Java Developer', 
    30: 'Front End Developer', 42: 'Network_Administrator', 
    46: 'Project_manager', 41: 'Network Security Engineer',
    50: 'Software_Developer', 51: 'Systems_Administrator', 
    54: 'Web Developer', 23: 'Database_Administrator', 
    21: 'Data Science', 33: 'HR', 5: 'Advocate', 
    6: 'Arts', 39: 'Mechanical Engineer', 
    49: 'Sales', 35: 'Health and fitness', 
    17: 'Civil Engineer', 13: 'Business Analyst', 
    48: 'SAP Developer', 7: 'Automation Testing', 
    28: 'Electrical Engineering', 43: 'Operations Manager', 
    24: 'DevOps Engineer', 44: 'PMO', 
    22: 'Database', 34: 'Hadoop', 
    27: 'ETL Developer', 25: 'DotNet Developer', 
    12: 'Blockchain', 53: 'Testing', 
    11: 'Backend Developer', 31: 'Full Stack Developer', 
    40: 'Mobile App Developer (iOS/Android)', 38: 'Machine Learning Engineer', 
    18: 'Cloud Engineer', 19: 'DESIGNER', 
    36: 'INFORMATION-TECHNOLOGY', 52: 'TEACHER', 
    10: 'BUSINESS-DEVELOPMENT', 32: 'HEALTHCARE', 
    1: 'AGRICULTURE', 9: 'BPO', 16: 'CONSULTANT', 
    20: 'DIGITAL-MEDIA', 3: 'AUTOMOBILE', 
    14: 'CHEF', 29: 'FINANCE', 
    2: 'APPAREL', 26: 'ENGINEERING', 
    0: 'ACCOUNTANT', 15: 'CONSTRUCTION', 
    45: 'PUBLIC-RELATIONS', 8: 'BANKING', 4: 'AVIATION'
}


# Utility Functions
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def extract_skills(text, job_skills):
    tokens = word_tokenize(text)
    return set(word for word, tag in pos_tag(tokens) if word in job_skills)

# App Layout
st.set_page_config(page_title="Resume Screening App", layout="wide")

# Sidebar
st.sidebar.title("Options")
selected_models = st.sidebar.multiselect("Select Models:", list(models.keys()))


# App Main
def main():
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:8px; border-radius:8px;">
            <h1 style="color:#2c3e50; text-align:center;">📄 Resume Insights and Optimization App</h1>
            <p style="text-align:center; font-size:16px;">Upload resumes, match skills, and tailor the resume!</p>
        </div>
        """, unsafe_allow_html=True
    )
    with st.expander("**Sample view**"):
        st.markdown("### Preview of the Original Dataset")
        dataset = pd.read_csv('data/data.csv')
        st.write(dataset.head())

    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description below:", height=200)

    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("Upload a resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])

    # Handle Resume Upload
    if uploaded_file is not None:
        with st.spinner("Processing resume..."):
            time.sleep(2)  # Simulate processing time

            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8")

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            st.success("Resume processed successfully!")

        # Predict Job Role
        if st.button("Categorize"):
            st.markdown("### Predictions:")
            results = {model: models[model].predict(input_features)[0] for model in selected_models}
            for model_name, category_id in results.items():
                st.markdown(
    f"""
    <div style="text-align: left; background-color: #f9f9f9; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <p style="font-size:18px; margin: 0;">The model used is {model_name}</p>
        <p style="font-size:18px; color: #2c3e50; margin: 0;">The predicted category is {category_mapping.get(category_id, "Unknown")}</p>
    </div>
    """,
    unsafe_allow_html=True
)
            # Skill Matching
            resume_skills = extract_skills(cleaned_resume, skills)
            job_skills = extract_skills(job_description, skills)
            common_skills = resume_skills.intersection(job_skills)
            match_percentage = len(common_skills) / len(job_skills) * 100 if job_skills else 0

            # Visualization
            skill_match_fig = px.bar(
                x=["Resume Skills", "Job Description Skills", "Matched Skills"],
                y=[len(resume_skills), len(job_skills), len(common_skills)],
                color=["Resume", "Job Description", "Matched"],
                title="Skill Matching Overview"
            )
            
            # Additional Visualization: Pie Chart
            pie_chart_fig = px.pie(
                values=[len(common_skills), len(job_skills) - len(common_skills)],
                names=["Matched Skills", "Unmatched Skills"],
                title="Skill Match Breakdown",
                color_discrete_sequence=px.colors.sequential.RdBu
            )

            # Additional Visualization: Heatmap or Radar Chart (if applicable)
            radar_fig = px.bar_polar(
                r=[len(resume_skills), len(job_skills), len(common_skills)],
                theta=["Resume Skills", "Job Description Skills", "Matched Skills"],
                color=["Resume", "Job Description", "Matched"],
                title="Skills Overview (Polar Chart)"
            )
            
            skill_data = pd.DataFrame({
                        "Category": ["Resume Skills", "Job Description Skills", "Matched Skills"],
                        "Count": [len(resume_skills), len(job_skills), len(common_skills)]
                    })
            
            # Skill Match Table
            st.dataframe(skill_data)

            # Expandable job skills
            with st.expander("**View Job Skills**"):
                st.write("**Job Skills:**")
                if job_skills:
                    for skill in sorted(job_skills):
                        st.markdown(f"- {skill}")
                else:
                    st.write("No skills detected in the job description.")

            # Expandable resume skills
            with st.expander("**View Resume Skills**"):
                st.write("**Resume Skills:**")
                if resume_skills:
                    for skill in sorted(resume_skills):
                        st.markdown(f"- {skill}")
                else:
                    st.write("No skills detected in the resume.")

            # Expandable matched skills
            with st.expander("**View Common Skills**"):
                st.write("**Common skills in both:**")
                if common_skills:
                    for skill in sorted(common_skills):
                        st.markdown(f"- {skill}")
                else:
                    st.write("No matched skills.")

            # Expandable unmatched skills
            with st.expander("**View Unmatched Skills**"):
                unmatched_skills = job_skills - resume_skills
                st.write("**Skills to be added:**")
                if unmatched_skills:
                    for skill in sorted(unmatched_skills):
                        st.markdown(f"- {skill}")
                else:
                    st.write("All job description skills are matched.")
                
            # Add Pie Chart and Polar Chart
            st.plotly_chart(skill_match_fig.update_layout(showlegend=False))
            st.plotly_chart(pie_chart_fig)
            st.plotly_chart(radar_fig)

            # Display skill matching results
            st.write(f"**Skill Match Percentage:** {match_percentage:.2f}%")
        else:
            st.write("**Please select at least one model.**")

if __name__ == "__main__":
    main()

