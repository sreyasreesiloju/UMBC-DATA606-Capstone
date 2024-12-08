# Resume Insights and Optimization

![Resume_Insights](resume_img.jpg)

## 1. Title and Author

* **Project Title**: Resume Insights and Optimization
* **Prepared for**: UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
* **Author Name**: Sreya Sree Siloju
* **GitHub Repository**: [GitHub Repo](https://github.com/sreya0299)
* **LinkedIn Profile**: [LinkedIn Profile](https://www.linkedin.com/in/sreya-sree-siloju-a29224149/)
* **PowerPoint Presentation**: PowerPoint Presentation
* **YouTube Video**: YouTube Video
  
---

## 2. Background

### What is it About?
This Project analyzes resumes to extract key details like skills, experience, and education. It offers suggestions for improving the resume by identifying missing skills, enhancing readability, and matching it to job descriptions. The system may also recommend relevant job positions and provide feedback to make resumes more ATS-compliant. It involves natural language processing (NLP) and machine learning to perform tasks like resume parsing, skill matching, and personalized job recommendations. It helps job seekers optimize their resumes for specific roles.

### Why Does it Matter?
1. **Improves Resume Shortlisting**: It allows candidates to tailor their resumes to specific job descriptions by recommending skills or qualifications they may lack, increasing their chances of aligning with job description.
2. **Reduces Time and Effort**: This automates the resume review process, allowing job seekers to quickly understand what improvements are needed while helping recruiters filter out unqualified candidates more efficiently.
3. **Enhances personalised Skill Development**: By identifying gaps in a resume and suggesting relevant skills or certifications, the system encourages continuous learning and professional development, aligning candidates with current market demands.

### Research Questions:
1. **Model Effectiveness**: How effectively does the model identify skill gaps between a candidate's resume and industry-standard requirements?
2. **Handing Multiple Resumes**: How well does the model handle various resume structures and formats when extracting relevant information?
3. **Skill Upgrade Recommendation**: How accurately can the model predict suitable job roles based on the skills and experience presented in the resumes?

---

## 3. Data

### Data Sources:
* **Data**: [Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)
* **Data Size**: 3.11 MB
* **Data Shape**: 962 X 2
* The dataset contains 962 rows and 2 columns 

### Representation of Dataset ?
* **Category**: Each row represents the category to which the resume belongs 
* **Resume**: Resume columns consists of resumes of each individual in the raw text.

### Data Dictionary:

| **Column Name** | **Data Type** | **Definition** | **Values**                            |
|-----------------|---------------|----------------|---------------------------------------|
| category        | string        | The field that the resume corresponds to, It indicates the type of job the resume is targeting.  | Java Developer, Operations Manager, Data Science, Testing etc. |
| Resume          | string        | The textual content of the resume contains the complete resume of an individual in raw text format, including their skills, education, experience, personal deatils etc.     | NA |

**Target/Label**:

1. **Category**: It is used as the **target** variable for classification tasks in which the model will attempt to predict the appropriate job category based on the content of the resume.

2. **Feature/Predictors**:
**Resume**: It contains the text of the resumes and is the **feature**. It is the input data that will be processed to extract relevant information (skills, experience, education etc.) to predict the target category.
The in-depth features that are extracted from each resume are:
* **Skills**:
  * **Technical Skills**: Programming languages (e.g., Python, Java), tools (e.g., Power BI, Tableau), and technologies (e.g., AWS, Azure).
  * **Soft Skills**: Communication, leadership, teamwork, problem-solving.
* **Education**:
  * **Degree Level**: Bachelor’s, Master’s, Ph.D.
  * **Field of Study**: Computer Science, Engineering, Finance, etc.
  * **Institutions**: Name of the universities or schools attended.
  * **Graduation Year**: Year the degree was completed.
* **Professional Experience**:
  * **Job Titles**: Positions held (e.g., Data Analyst, Software Engineer).
  * **Years of Experience**: The total number of years or specific time periods of work.
  * **Companies**: Names of companies or organizations where the individual worked.
  * **Job Responsibilities**: The tasks performed in each role.
* **Keywords and Phrases**:
  * Specific industry-related terms that are commonly found in job descriptions and are required for matching (e.g., "Machine Learning," "Data Analysis," "Project Management", "Big Data").

**Packages and Technologies**:
* Backend Development using **Python** programming language
* Text is converted into numericals using **TFID Vectorizer**, **KNeighborsClassifier** and **OneVsRestClassifier** are used for classification of the resume into it's category as there are multiple classes in the target variable.
* Machine Learning Packages are used for model training and model predicting to which the resume belongs to which helps in identifyinng the skills responsible for each category. 
* By using **NLTK**, the resume text is processed and cleaned, ensuring better input for further tasks like feature extraction and classification.
* **NLP** concepts like Named Entity Recognition (NER) are used for identifying the key words from the resume like skills, experience, technologies worked on etc.
* **Streamlit** is used for creating a web interface for evaluating the resume with provided job description.
---
