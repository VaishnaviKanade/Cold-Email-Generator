import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

    # ------------------ JOB EXTRACTION ------------------
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The text is from a job/careers page.
            Extract job details and return VALID JSON only.

            REQUIRED KEYS:
            - role
            - experience
            - skills
            - description

            ### VALID JSON (NO EXTRA TEXT):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            res = parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse job details.")

        return res if isinstance(res, list) else [res]

    # ------------------ EMAIL GENERATION ------------------
    def write_mail(self, job, resume_text, tone, length):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### RESUME CONTENT:
            {resume}

            ### TONE:
            {tone}

            ### EMAIL LENGTH:
            {length}

            ### INSTRUCTIONS (VERY IMPORTANT):
            - You are the JOB APPLICANT, not a company
            - Extract the candidate's FULL NAME from the resume
            - Use the extracted name ONLY in the email closing
            - If name is unclear, close with "Best regards," only
            - Do NOT invent any name
            - Write in simple, natural, human language

            ### EMAIL FORMAT RULES:
            - Add a clear subject line
            - Greeting must be: "Dear Hiring Manager,"
            - Break content into clean paragraphs (2–3 lines each)
            - Match resume skills with job requirements
            - Mention resume politely
            - No emojis
            - No preamble or explanation

            ### REQUIRED STRUCTURE:
            Subject: ...

            Dear Hiring Manager,

            Paragraph 1: Introduction and role interest
            Paragraph 2: Skills aligned with job
            Paragraph 3: Projects / experience
            Paragraph 4: Polite closing

            Best regards,
            <Candidate Name from Resume>
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "resume": resume_text,
            "tone": tone,
            "length": length
        })

        return res.content


if __name__ == "__main__":
    print("GROQ_API_KEY Loaded:", bool(os.getenv("GROQ_API_KEY")))
