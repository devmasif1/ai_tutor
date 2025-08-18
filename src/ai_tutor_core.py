import uuid
import requests
import json
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch  

from src.knowledge_base import MongoKnowledgeBase 
from src.vector_visualizer import VectorVisualizer
from src.config import MODEL

import json

class QuizGenerator:
    """Handles quiz generation and evaluation with upfront subject/topic specification"""
    
    def __init__(self, model, knowledge_base):
        self.model = model
        self.kb = knowledge_base
    
    def generate_quiz(self, subject, topic, marks=10, quiz_type="Mixed"):
        """Generate quiz with subject and topic specified upfront"""
        
        # Create the main prompt combining subject, topic
        main_prompt = f"Generate a quiz on {subject} - {topic}"
        
        # Get context from knowledge base if available
        context = ""
        if self.kb and self.kb.vectorstore:
            try:
                search_query = f"{subject} {topic}"
                relevant_docs = self.kb.similarity_search_with_user_filter(search_query, k=10)
                if relevant_docs:
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
            except Exception as e:
                print(f"[WARNING] Could not retrieve context for quiz: {e}")
        
        # Create enhanced quiz generation prompt
        quiz_prompt = f"""
You are an AI Quiz Generator. Generate a conceptual quiz based on the following requirements:

Subject: {subject}
Topic: {topic}
User Request: {main_prompt}
Marks Required: {marks}
Quiz Type: {quiz_type}

Available Context from User's Documents:
{context if context else "No specific context available - use general knowledge"}

Instructions:
1. Generate CONCEPTUAL questions that test understanding, not memorization
2. If coding topic: include dry run problems, coding challenges, and algorithm analysis
3. Make questions from the context, one you get the general overview of context 
and what's been the context it is about , you can generate question out of book.
4. Focus on "why" and "how" rather than "what" - test deep understanding
5. Include scenario-based and application questions
6. Make questions challenging but fair
7. Make questions marks as per type (like mcq 1 mark)
8.make sure you follow sequence in case of mixed quiz like mcqs first , then something and then something

Return quiz in this JSON format:
{{
    "needs_clarification": false,
    "quiz": {{
        "title": "Quiz on {topic}",
        "subject": "{subject}",
        "topic": "{topic}",
        "total_marks": {marks},
        "questions": [
            {{
                "question": "Question text",
                "type": "MCQ/Short Answer/Fill-in-blank/Coding/Dry-Run",
                "marks": 1,
                "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                "correct_answer": "answer",
                "explanation": "detailed explanation"
            }}
        ]
    }}
}}

Only return the JSON, nothing else.
"""
        
        try:
            response = self.model.invoke([{"role": "user", "content": quiz_prompt}])
            quiz_data = self._parse_json_response(response.content)
            
            # Ensure subject and topic are included in the quiz data
            if "quiz" in quiz_data and quiz_data["quiz"]:
                quiz_data["quiz"]["subject"] = subject
                quiz_data["quiz"]["topic"] = topic
            
            return quiz_data
        except Exception as e:
            return {
                "needs_clarification": False,
                "quiz": {
                    "title": f"Quiz on {topic}",
                    "subject": subject,
                    "topic": topic,
                    "total_marks": marks,
                    "questions": [],
                    "error": f"Failed to generate quiz: {str(e)}"
                }
            }
    
    def evaluate_quiz_and_get_metadata(self, quiz_data, user_answers):
        """Evaluate quiz and return results with subject/topic already included"""
        
        questions = quiz_data["quiz"]["questions"]
        total_marks = quiz_data["quiz"]["total_marks"]
        subject = quiz_data["quiz"]["subject"]  
        topic = quiz_data["quiz"]["topic"]      
        
        scored_marks = 0
        detailed_feedback = []
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(f"q_{i}", "").strip()
            correct_answer = question["correct_answer"].strip()
            question_marks = question["marks"]
            
            # Evaluate based on question type
            marks_awarded = self._evaluate_question(question, user_answer, correct_answer, question_marks)
            scored_marks += marks_awarded
            
            # Generate feedback
            feedback = self._generate_question_feedback(marks_awarded, question_marks, question["explanation"])
            
            detailed_feedback.append({
                "question": question["question"],
                "type": question.get("type", "Unknown"),
                "your_answer": user_answer,
                "correct_answer": correct_answer,
                "marks_awarded": marks_awarded,
                "max_marks": question_marks,
                "feedback": feedback
            })
        
        # Generate overall feedback
        percentage = (scored_marks / total_marks) * 100 if total_marks > 0 else 0
        overall_feedback = self._generate_overall_feedback(percentage, topic)
        
        return {
            "scored_marks": scored_marks,
            "total_marks": total_marks,
            "percentage": round(percentage, 2),
            "overall_feedback": overall_feedback,
            "detailed_feedback": detailed_feedback,
            "subject": subject,  # No need to ask user again!
            "topic": topic      # No need to ask user again!
        }
    
    def _evaluate_question(self, question, user_answer, correct_answer, question_marks):
        """Fixed evaluation with proper empty answer handling"""
        
        question_type = question.get("type", "Short Answer")
        
        # CRITICAL: Check for empty answers first
        if not user_answer or not user_answer.strip():
            return 0  # No marks for empty answers
        
        user_answer = user_answer.strip()
        
        if question_type == "MCQ":
            return question_marks if user_answer.upper() == correct_answer.upper() else 0
        
        else:
            # Use AI evaluation for non-MCQ questions with stricter prompt
            eval_prompt = f"""
            Question Type: {question_type}
            Question: {question["question"]}
            Correct Answer: {correct_answer}
            User Answer: {user_answer}
            
            STRICT EVALUATION RULES:
            - If user answer is empty, incomplete, or clearly wrong: award 0 marks
            - Only award full marks if answer demonstrates complete understanding
            - Award partial marks only if core concepts are correct but incomplete
            - Be conservative with scoring - when in doubt, award fewer marks
            
            Evaluate considering:
            - Correctness of core concepts (must be present)
            - Completeness of answer
            - For coding: syntax and logic correctness
            - For explanations: key points covered
            
            Return ONLY a number between 0 and {question_marks}. Be strict.
            """
            
            try:
                eval_response = self.model.invoke([{"role": "user", "content": eval_prompt}])
                awarded_marks = float(eval_response.content.strip())
                
                # Ensure marks are within bounds
                awarded_marks = max(0, min(awarded_marks, question_marks))
                
                # Additional validation: if answer is too short for complex questions, limit marks
                if len(user_answer) < 10 and question_marks > 2:
                    awarded_marks = min(awarded_marks, question_marks * 0.3)
                
                return awarded_marks
                
            except Exception as e:
                print(f"[ERROR] AI evaluation failed: {e}")
                # Fallback to conservative manual check
                if user_answer.lower().strip() == correct_answer.lower().strip():
                    return question_marks
                elif any(word in user_answer.lower() for word in correct_answer.lower().split() if len(word) > 3):
                    return question_marks * 0.3  # Minimal partial credit
                else:
                    return 0

    def evaluate_quiz_and_get_metadata(self, quiz_data, user_answers):
        """Fixed evaluation with better validation"""
        
        questions = quiz_data["quiz"]["questions"]
        total_marks = quiz_data["quiz"]["total_marks"]
        subject = quiz_data["quiz"].get("subject", "Unknown")
        topic = quiz_data["quiz"].get("topic", "Unknown")
        
        scored_marks = 0
        detailed_feedback = []
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(f"q_{i}", "").strip()
            correct_answer = question["correct_answer"].strip()
            question_marks = question["marks"]
            
            # Evaluate with fixed logic
            marks_awarded = self._evaluate_question(question, user_answer, correct_answer, question_marks)
            scored_marks += marks_awarded
            
            # Generate feedback with empty answer detection
            feedback = self._generate_question_feedback_fixed(
                user_answer, marks_awarded, question_marks, question["explanation"]
            )
            
            detailed_feedback.append({
                "question": question["question"],
                "type": question.get("type", "Unknown"),
                "your_answer": user_answer if user_answer else "No answer provided",
                "correct_answer": correct_answer,
                "marks_awarded": marks_awarded,
                "max_marks": question_marks,
                "feedback": feedback
            })
        
        # Ensure total scored marks don't exceed total possible marks
        scored_marks = min(scored_marks, total_marks)
        
        # Generate overall feedback
        percentage = (scored_marks / total_marks) * 100 if total_marks > 0 else 0
        overall_feedback = self._generate_overall_feedback_fixed(percentage, topic, len([f for f in detailed_feedback if not f["your_answer"] or f["your_answer"] == "No answer provided"]))
        
        return {
            "scored_marks": scored_marks,
            "total_marks": total_marks,
            "percentage": round(percentage, 2),
            "overall_feedback": overall_feedback,
            "detailed_feedback": detailed_feedback,
            "subject": subject,
            "topic": topic
        }

    def _generate_question_feedback_fixed(self, user_answer, marks_awarded, max_marks, explanation):
        """Generate feedback with proper empty answer handling"""
        
        if not user_answer or user_answer == "No answer provided":
            return f"❌ No answer provided. {explanation}"
        
        percentage = (marks_awarded / max_marks) * 100 if max_marks > 0 else 0
        
        if percentage == 100:
            return f"✅ Excellent! {explanation}"
        elif percentage >= 75:
            return f"🟢 Very Good! {explanation}"
        elif percentage >= 50:
            return f"🟡 Partially Correct. {explanation}"
        elif percentage > 0:
            return f"🟠 Some understanding shown, but incomplete. {explanation}"
        else:
            return f"❌ Incorrect. {explanation}"

    def _generate_overall_feedback_fixed(self, percentage, topic, empty_answers_count):
        """Generate overall feedback considering empty answers"""
        
        feedback_base = ""
        if empty_answers_count > 0:
            feedback_base = f"⚠️ Note: {empty_answers_count} questions were left unanswered. "
        
        if percentage >= 90:
            return f"{feedback_base}🏆 Outstanding mastery of {topic}! You have excellent conceptual understanding."
        elif percentage >= 80:
            return f"{feedback_base}🎯 Great work on {topic}! Strong conceptual grasp with minor gaps."
        elif percentage >= 70:
            return f"{feedback_base}👍 Good understanding of {topic}. Focus on strengthening weaker concepts."
        elif percentage >= 60:
            return f"{feedback_base}📚 Fair grasp of {topic}. Review core concepts and practice more problems."
        elif percentage >= 40:
            return f"{feedback_base}📖 Basic understanding of {topic}. Significant review needed for key concepts."
        else:
            return f"{feedback_base}🔄 {topic} needs comprehensive review. Focus on fundamentals and attempt all questions."
        
    def _parse_json_response(self, response_text):
        """Parse JSON from AI response"""
        try:
            # Try to find JSON in the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
            else:
                return json.loads(response_text)
        except Exception as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            return {
                "needs_clarification": False,
                "quiz": {
                    "title": "Parse Error",
                    "subject": "Unknown",
                    "topic": "Unknown", 
                    "total_marks": 0,
                    "questions": [],
                    "error": f"Could not parse quiz data: {str(e)}"
                }
            }

class AITutorCore:

    def __init__(self, user_id: str = None, username: str = None, backend_url: str = "http://localhost:8000", token: str = None):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        
        self.user_id = user_id
        self.username = username
        self.backend_url = backend_url
        self.token = token
        
        # Initialize MongoDB-based Knowledge Base with per-user collection
        try:
            self.KB = MongoKnowledgeBase(
                user_id=self.user_id,
                username=self.username
            )
            print(f"[INFO] MongoDB Knowledge Base initialized for user: {self.username}")
            print(f"[INFO] User collection: {self.KB.collection_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize MongoDB Knowledge Base: {e}")
            self.KB = None
        
        # Initialize workflow components
        self.memory = MemorySaver()
        self.workflow = StateGraph(state_schema=MessagesState)
        self.model = ChatOpenAI(temperature=0.7, model=MODEL)
        self.thread_id = str(uuid.uuid4())
        self.history = []
        self.visualizer = None
        
        # Initialize Quiz Generator
        self.quiz_generator = QuizGenerator(self.model, self.KB)
        
        self._setup_workflow()
        self._initialize_visualizer_if_needed()

    def _setup_workflow(self):
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self._call_model)
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _call_model(self, state: MessagesState):
        response = self.model.invoke(state["messages"])
        return {"messages": response}
    
    def _initialize_visualizer_if_needed(self):
        """Initialize visualizer if knowledge base exists"""
        if self.KB and self.KB.vectorstore is not None:  
            try:
                self.visualizer = VectorVisualizer(self.KB.vectorstore)
                print(f"[INFO] Vector visualizer initialized for user: {self.username}")
            except Exception as e:
                print(f"[WARNING] Could not initialize visualizer for user {self.username}: {e}")
        else:
            print(f"[INFO] No vectorstore available for user {self.username}")
    
    def chat_with_tutor(self, message, history):
        """Simple chat function - NO session tracking"""
        self.history = history if history else []
        
        # Get context from knowledge base if available
        context = ""
        retrieval_info = ""
        
        if self.KB and self.KB.vectorstore:
            try:
                relevant_docs = self.KB.similarity_search_with_user_filter(message, k=5)
                if relevant_docs:
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    retrieval_info = f"Based on your uploaded documents (User: {self.username})"
                else:
                    retrieval_info = "No relevant documents found in your knowledge base"
            except Exception as e:
                print(f"[ERROR] Retrieval failed for user {self.username}: {e}")
                retrieval_info = "Retrieval failed"
        else:
            retrieval_info = f"No knowledge base available for user {self.username}"

        # Build chat history for context
        chat_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.history
        ) if self.history else "No previous conversation"

        # Create STRICT system prompt - only respond from documents
        if context.strip():
            system_prompt = f"""
    You are AI Tutor Assistant for User: {self.username}

    🎯 STRICT INSTRUCTIONS:
    - You MUST only answer based on the provided document context below
    - If the answer is not in the documents, say "I don't have information about this in your uploaded documents"
    - You may use general knowledge only to enhance your answer when:
        1) You have context from documents but need to add examples not provided in the context
        2) The user requests different examples and the context only provides one
        3)You need to clarify or explain concepts mentioned in the documents
    - Always be an intelligent and helpful assistant

    Context from uploaded documents:
    {context}

    Chat History:
    {chat_history}

    Question: {message}

    Answer:
    """
        else:
            system_prompt = f"""
    You are AI Tutor Assistant for User: {self.username}

    🎯 STATUS: {retrieval_info}

    I don't have access to any relevant documents in your knowledge base to answer this question. 

    To get answers from your documents, please:
    1. Upload relevant documents to your knowledge base
    2. Ask questions related to the uploaded content

    Chat History:
    {chat_history}

    Question: {message}

    Response: I don't have information about "{message}" in your uploaded documents. Please upload relevant documents first, or ask questions about content you've already uploaded.
    """

        messages = [SystemMessage(content=system_prompt)]
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        try:
            for event in self.app.stream({"messages": messages}, config, stream_mode="values"):
                last_msg = event["messages"][-1]
                answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            
            return answer
            
        except Exception as e:
            print(f"[ERROR] Chat generation failed for user {self.username}: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def generate_quiz(self, subject, topic, marks=10, quiz_type="Mixed"):
        """Generate quiz with subject and topic"""
        return self.quiz_generator.generate_quiz(subject, topic, marks, quiz_type)
        
    def evaluate_quiz_and_get_metadata(self, quiz_data, user_answers):
        """Evaluate quiz and return results with metadata - delegates to quiz_generator"""
        return self.quiz_generator.evaluate_quiz_and_get_metadata(quiz_data, user_answers)

    def save_quiz_report(self, evaluation):
        """Save quiz report to backend"""
        if not self.token:
            print("[WARNING] No token available to save quiz report")
            return
        
        try:
            report_data = {
                "topic": evaluation["topic"],
                "subject": evaluation["subject"],
                "total_marks": evaluation["total_marks"],
                "scored_marks": evaluation["scored_marks"],
                "feedback": evaluation["overall_feedback"],
                "quiz_data": {
                    "questions": evaluation.get("detailed_feedback", []),
                    "percentage": evaluation["percentage"]
                }
            }
            
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.post(
                f"{self.backend_url}/save_quiz_report",
                json=report_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"[INFO] Quiz report saved for user {self.username}")
            else:
                print(f"[ERROR] Failed to save quiz report: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Could not save quiz report: {e}")
    
    
    def evaluate_quiz_and_save_report(self, quiz_data, user_answers, topic, subject):
        """Evaluate quiz and save report to backend"""
        
        evaluation = self.quiz_generator.evaluate_quiz(quiz_data, user_answers)
        
        report_data = {
            "topic": topic,
            "subject": subject,
            "total_marks": evaluation["total_marks"],
            "scored_marks": evaluation["scored_marks"],
            "feedback": evaluation["overall_feedback"],
            "quiz_data": {
                "quiz_title": quiz_data["quiz"]["title"],
                "questions": quiz_data["quiz"]["questions"],
                "user_answers": user_answers,
                "detailed_feedback": evaluation["detailed_feedback"],
                "percentage": evaluation["percentage"]
            }
        }
        
        if self.token:
            try:
                headers = {"Authorization": f"Bearer {self.token}"}
                response = requests.post(
                    f"{self.backend_url}/save_quiz_report",
                    json=report_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"[INFO] Quiz report saved for user {self.username}")
                else:
                    print(f"[ERROR] Failed to save quiz report: {response.text}")
                    
            except Exception as e:
                print(f"[ERROR] Could not save quiz report: {e}")
        
        return evaluation
    
    def get_subject_report(self, subject):
        """Get comprehensive report for a subject"""
        if not self.token:
            return "Please login to view reports"
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                f"{self.backend_url}/generate_subject_report/{subject}",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get report: {response.text}"}
                
        except Exception as e:
            return {"error": f"Could not fetch report: {str(e)}"}
    
    def get_all_subjects(self):
        """Get all subjects user has taken quizzes on"""
        if not self.token:
            return []
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                f"{self.backend_url}/get_all_subjects",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()["subjects"]
            else:
                return []
                
        except Exception as e:
            return []
    
    # Existing methods (unchanged)
    def add_content_to_kb(self, path):
        """Add content to knowledge base from file, folder, or ZIP"""
        if not self.KB:
            return f"Error: Knowledge base not initialized for user {self.username}"
        
        try:
            result = self.KB.add_content(path)
            print(f"[INFO] Content added to KB for user {self.username}")

            if self.KB.vectorstore is not None:
                try:
                    self.visualizer = VectorVisualizer(self.KB.vectorstore)
                    print("[INFO] Vector visualizer refreshed after adding content.")
                except Exception as e:
                    print(f"[WARNING] Could not refresh visualizer: {e}")
            
            return result
            
        except Exception as e:
            return f"Error adding content for user {self.username}: {str(e)}"
    
    def show_visualization(self, vis_type):
        """Show vector visualization"""
        if self.KB.vectorstore is None:
            return "No knowledge base available. Please add some content first."
            
        if not self.visualizer:
            try:
                self.visualizer = VectorVisualizer(self.KB.vectorstore)
            except Exception as e:
                return f"Could not initialize visualizer: {str(e)}"
        
        try:
            if vis_type == "2D Visualization":
                self.visualizer.visualize_2d()
                return "2D visualization displayed"
            elif vis_type == "3D Visualization":
                self.visualizer.visualize_3d()
                return "3D visualization displayed"
            else:
                return "Please select a visualization type (2D or 3D)"
        except Exception as e:
            return f"Visualization failed: {str(e)}"

    def get_kb_status(self):
        """Get current knowledge base status"""
        if not self.KB:
            return f"Knowledge base not initialized for user {self.username}"
        
        try:
            return self.KB.investigate_vectors()
        except Exception as e:
            return f"Error getting KB status for user {self.username}: {str(e)}"

    def clear_knowledge_base(self):
        """Clear the entire knowledge base for this user"""
        if not self.KB:
            return f"Knowledge base not initialized for user {self.username}"
        
        try:
            result = self.KB.clear_knowledge_base()
            print(f"[INFO] Knowledge base cleared for user {self.username}")
            
            # Reinitialize visualizer after clearing
            self._initialize_visualizer_if_needed()
            
            return result
        except Exception as e:
            return f"Error clearing KB for user {self.username}: {str(e)}"

    def get_directory_summary(self, path):
        """Get directory structure summary"""
        if not self.KB:
            return f"Knowledge base not initialized for user {self.username}"
        
        return self.KB.get_directory_summary(path)
    
    def set_user(self, user_id, username=None):
        """Set user and reinitialize knowledge base and visualizer"""
        try:
            # Close existing connections
            self.close()
            
            # Set new user info
            self.user_id = user_id
            self.username = username or user_id
            
            self.KB = MongoKnowledgeBase(user_id=user_id, username=self.username)
            print(f"[INFO] MongoDB Knowledge Base reinitialized for user: {self.username}")
            print(f"[INFO] New user collection: {self.KB.collection_name}")
            
            self._initialize_visualizer_if_needed()
            
            # Reset other user-specific data
            self.history = []
            self.thread_id = str(uuid.uuid4())  # New thread for new user
            
            return f"Successfully switched to user: {self.username} (Collection: {self.KB.collection_name})"
            
        except Exception as e:
            print(f"[ERROR] Failed to switch user to {self.username}: {e}")
            return f"Error switching to user {self.username}: {str(e)}"
        
    def close(self):
        """Close knowledge base connection"""
        if self.KB:
            try:
                self.KB.close()
                print(f"[INFO] Closed KB connection for user {self.username}")
            except Exception as e:
                print(f"[WARNING] Error closing KB for user {self.username}: {e}")

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()