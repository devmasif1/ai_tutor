import gradio as gr
import requests
import json
import uuid
from datetime import datetime
from src.ai_tutor_core import AITutorCore

class AITutorApp:
    def __init__(self):
        self.core = None  # Will be initialized when user logs in
        self.backend_url = "http://localhost:8000"
        self.token = None
        self.current_user = None
        self.current_user_id = None
        self.current_quiz = None 
        self.demo = self._create_interface()

    def register_user(self, username, password, email):
        try:
            response = requests.post(f"{self.backend_url}/register", json={
                "username": username,
                "password": password,
                "email": email
            })
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["token"]
                self.current_user = data["user"]
                self.current_user_id = data["user"]["id"]
                
                # Initialize core with user ID and token
                try:
                    self.core = AITutorCore(
                        user_id=self.current_user_id,
                        username=username,
                        backend_url=self.backend_url,
                        token=self.token
                    )
                    print(f"[INFO] Initialized AI Tutor for user: {username} (ID: {self.current_user_id})")
                except Exception as e:
                    print(f"[ERROR] Failed to initialize core for user {username}: {e}")
                    return f"✅ Registered but failed to initialize personal AI: {str(e)}", gr.update(), gr.update()
                
                return f"✅ Registered successfully! Welcome, {username}!", gr.update(visible=False), gr.update(visible=True)
            else:
                return f"❌ Registration failed: {response.json().get('detail', 'Unknown error')}", gr.update(), gr.update()
        except Exception as e:
            return f"❌ Connection error: {str(e)}", gr.update(), gr.update()

    def login_user(self, username, password):
        try:
            response = requests.post(f"{self.backend_url}/login", json={
                "username": username,
                "password": password
            })
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["token"]
                self.current_user = data["user"]
                self.current_user_id = data["user"]["id"]
                
                # Initialize core with user ID and token
                try:
                    self.core = AITutorCore(
                        user_id=self.current_user["id"],
                        username=self.current_user["username"],
                        backend_url=self.backend_url,
                        token=self.token
                    )
                    print(f"[INFO] Initialized AI Tutor for user: {username} (ID: {self.current_user_id})")
                except Exception as e:
                    print(f"[ERROR] Failed to initialize core for user {username}: {e}")
                    return f"✅ Logged in but failed to initialize personal AI: {str(e)}", gr.update(), gr.update()
                
                return f"✅ Welcome back, {username}!", gr.update(visible=False), gr.update(visible=True)
            else:
                return f"❌ Login failed: {response.json().get('detail', 'Invalid credentials')}", gr.update(), gr.update()
        except Exception as e:
            return f"❌ Connection error: {str(e)}", gr.update(), gr.update()

    def logout_user(self):
        # Clean up core resources
        if self.core:
            try:
                self.core.close()
                print(f"[INFO] Closed AI Tutor for user: {self.current_user_id}")
            except Exception as e:
                print(f"[WARNING] Error closing core: {e}")
        
        self.token = None
        self.current_user = None
        self.current_user_id = None
        self.current_quiz = None
        self.core = None
        
        return "👋 Logged out successfully!", gr.update(visible=True), gr.update(visible=False)

    def chat_with_tutor(self, message, history):
        """Simple chat with tutor - NO session tracking"""
        if not self.token or not self.core:
            return "⚠️ Please login first to use the AI Tutor."
        
        try:
            ai_response = self.core.chat_with_tutor(message, history)
            return ai_response
        except Exception as e:
            print(f"Chat error for user {self.current_user_id}: {e}")
            return f"AI Error for user {self.current_user_id}: {str(e)}"

    # Updated Quiz-related methods
    def generate_quiz(self, subject, topic, marks, quiz_type):
        """Generate quiz with upfront subject/topic specification"""
        if not self.token or not self.core:
            return "⚠️ Please login first to generate quiz.", gr.update(visible=False)
        
        if not subject.strip() or not topic.strip():
            return "⚠️ Please specify both subject and topic.", gr.update(visible=False)
        
        try:
            quiz_data = self.core.generate_quiz(subject.strip(), topic.strip(), marks, quiz_type )
            
            if quiz_data.get("needs_clarification", False):
                return quiz_data.get("clarification_message", "Please provide more details"), gr.update(visible=False)
            
            # Store quiz data for evaluation later
            self.current_quiz = quiz_data
            
            # Format quiz for display
            quiz_html = self._format_quiz_for_display(quiz_data)
            
            return (
                f"✅ Quiz on {topic} ({subject}) generated successfully! Please answer below.", 
                gr.update(value=quiz_html, visible=True)
            )
                    
        except Exception as e:
            return f"❌ Error generating quiz: {str(e)}", gr.update(visible=False)

    def _format_quiz_for_display(self, quiz_data):
        """Format quiz as HTML for display with dark theme"""
        if "error" in quiz_data.get("quiz", {}):
            return f"<div style='color: #ff6b6b; background: #2d3748; padding: 15px; border-radius: 8px;'>Error: {quiz_data['quiz']['error']}</div>"
        
        quiz = quiz_data["quiz"]
        html = f"""
        <div style='padding: 20px; border: 2px solid #4a5568; border-radius: 12px; background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <h3 style='color: #63b3ed; margin-bottom: 15px;'>📝 {quiz['title']}</h3>
            <p style='color: #a0aec0;'><strong>Subject:</strong> {quiz.get('subject', 'N/A')} | <strong>Topic:</strong> {quiz.get('topic', 'N/A')}</p>
            <p style='color: #a0aec0; margin-bottom: 20px;'><strong>Total Marks:</strong> {quiz['total_marks']}</p>
            <hr style='border: 1px solid #4a5568; margin: 20px 0;'>
        """
        
        for i, question in enumerate(quiz["questions"]):
            html += f"""
            <div style='margin: 20px 0; padding: 20px; border: 1px solid #4a5568; border-radius: 8px; background: rgba(74, 85, 104, 0.3);'>
                <p style='color: #e2e8f0; font-weight: bold; margin-bottom: 15px;'>Question {i+1}: {question['question']} <span style='color: #fbb6ce;'>({question['marks']} marks)</span></p>
            """
            
            if question.get("type") == "MCQ":
                html += "<div style='margin: 15px 0; color: #cbd5e0;'>"
                for option in question.get("options", []):
                    html += f"<div style='margin: 8px 0; padding: 5px 0;'>• {option}</div>"
                html += "</div>"
                html += f"<input type='text' id='q_{i}' placeholder='Enter your answer (A/B/C/D)' style='width: 100%; padding: 12px; margin-top: 15px; border: 1px solid #4a5568; border-radius: 6px; background: #1a202c; color: white;'/>"
            else:
                html += f"<textarea id='q_{i}' placeholder='Enter your answer here...' style='width: 100%; height: 100px; padding: 12px; margin-top: 15px; border: 1px solid #4a5568; border-radius: 6px; background: #1a202c; color: white; resize: vertical;'></textarea>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def submit_quiz_answers(self, *answers):
        """Submit quiz answers - no need for subject/topic input anymore"""
        if not self.current_quiz or not self.core:
            return "❌ No active quiz found. Please generate a quiz first."
        
        try:
            # Convert answers to dictionary
            user_answers = {}
            for i, answer in enumerate(answers):
                if answer and answer.strip():
                    user_answers[f"q_{i}"] = answer.strip()
            
            # Evaluate quiz using the corrected method
            evaluation = self.core.evaluate_quiz_and_get_metadata(self.current_quiz, user_answers)
            
            # Save the report (assuming this method exists in core)
            try:
                self.core.save_quiz_report(evaluation)
            except Exception as e:
                print(f"[WARNING] Could not save quiz report: {e}")
            
            # Format evaluation results
            result_text = f"""
📊 **Quiz Results for {evaluation.get('topic', 'Unknown')} ({evaluation.get('subject', 'Unknown')})**

**Score:** {evaluation['scored_marks']}/{evaluation['total_marks']} ({evaluation['percentage']:.1f}%)

**Overall Feedback:** {evaluation['overall_feedback']}

**Detailed Results:**
"""
            
            for feedback in evaluation['detailed_feedback']:
                result_text += f"""
**Q: {feedback['question']}**
- **Your Answer:** {feedback['your_answer']}
- **Correct Answer:** {feedback['correct_answer']}
- **Marks:** {feedback['marks_awarded']}/{feedback['max_marks']}
- {feedback['feedback']}
---
"""
            
            result_text += f"\n✅ Quiz report automatically saved to your profile!"
            
            # Clear current quiz
            self.current_quiz = None
            
            return result_text
            
        except Exception as e:
            return f"❌ Error submitting quiz: {str(e)}"

    def get_subject_report(self, subject):
        """Get comprehensive report for a subject"""
        if not self.core:
            return "⚠️ Please login first."
        
        if not subject:
            return "⚠️ Please specify a subject."
        
        try:
            report_data = self.core.get_subject_report(subject)
            
            if "error" in report_data:
                return f"❌ {report_data['error']}"
            
            # Format report
            report_text = f"""
📊 **Comprehensive Report for {subject}**

📈 **Overall Performance:**
- Total Quizzes Taken: {report_data['total_quizzes']}
- Average Score: {report_data['average_percentage']:.1f}%
- Total Marks: {report_data['total_marks_scored']}/{report_data['total_marks_possible']}

🎯 **Topics Covered:** {', '.join(report_data['topics_covered'])}

📚 **Recent Performance:**
"""
            
            for performance in report_data['recent_performance']:
                date = performance['date'][:10] if isinstance(performance['date'], str) else str(performance['date'])[:10]
                report_text += f"• {performance['topic']}: {performance['percentage']:.1f}% ({date})\n"
            
            return report_text
            
        except Exception as e:
            return f"❌ Error generating report: {str(e)}"

    def refresh_subjects_dropdown(self):
        """Refresh the subjects dropdown"""
        if not self.core:
            return gr.update(choices=[])
        
        try:
            subjects = self.core.get_all_subjects()
            return gr.update(choices=subjects if subjects else ["No subjects found"])
        except:
            return gr.update(choices=["Error loading subjects"])

    # Existing methods (unchanged)
    def get_kb_status(self):
        """Get current knowledge base status"""
        if not self.core:
            return f"⚠️ AI Tutor not initialized. Please login first."
        
        return self.core.get_kb_status()

    def add_files_to_kb(self, files):
        """Add uploaded files to knowledge base"""
        if not self.core:
            return "⚠️ Please login first to upload files."
        
        if not files:
            return "⚠️ No files selected."
        
        try:
            results = []
            for file in files:
                if hasattr(file, 'name'):
                    file_path = file.name
                else:
                    file_path = str(file)
                
                result = self.core.add_content_to_kb(file_path)
                results.append(f"📄 {file_path}:\n{result}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"❌ Error adding files for user {self.current_user_id}: {str(e)}"

    def show_visualization(self, vis_type):
        """Show vector visualization"""
        return self.core.show_visualization(vis_type)

    def clear_knowledge_base(self):
        """Clear the entire knowledge base"""
        if not self.core:
            return "⚠️ Please login first."
        
        try:
            result = self.core.clear_knowledge_base()
            return f"✅ {result}"
        except Exception as e:
            return f"❌ Error clearing knowledge base for user {self.current_user_id}: {str(e)}"

    def _create_interface(self):
        # Custom dark theme
        dark_theme = gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ).set(
            body_background_fill="*neutral_950",
            body_text_color="*neutral_100",
            button_primary_background_fill="*primary_600",
            button_primary_text_color="white",
            input_background_fill="*neutral_800",
            input_border_color="*neutral_700",
            block_background_fill="*neutral_900",
            block_border_color="*neutral_700",
            panel_background_fill="*neutral_800"
        )
        
        with gr.Blocks(title="AI Tutor - Quiz-Based Learning Assistant", theme=dark_theme) as demo:
            gr.Markdown("# 🤖 AI Tutor - Quiz-Based Learning Assistant")
            
            # Authentication section
            auth_section = gr.Column(visible=True)
            main_section = gr.Column(visible=False)
            
            with auth_section:
                gr.Markdown("## 🔐 Login or Register to Start Learning")
                gr.Markdown("*Take quizzes and track your progress with personalized reports*")
                
                with gr.Tabs():
                    with gr.Tab("Login"):
                        login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                        login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                        login_btn = gr.Button("🚀 Login", variant="primary")
                        
                    with gr.Tab("Register"):
                        reg_username = gr.Textbox(label="Username", placeholder="Choose a username")
                        reg_email = gr.Textbox(label="Email", placeholder="Enter your email")
                        reg_password = gr.Textbox(label="Password", type="password", placeholder="Create a password")
                        register_btn = gr.Button("📝 Register", variant="primary")
                
                auth_message = gr.Textbox(label="Status", interactive=False)
            
            # Main application
            with main_section:
                with gr.Row():
                    gr.Markdown(f"### 🎓 Welcome to your Personal AI Learning Session!")
                    logout_btn = gr.Button("🚪 Logout", size="sm")
                
                with gr.Tabs():
                    # Chat Tab
                    with gr.Tab("💬 Chat"):
                        chatbot = gr.Chatbot(
                            type="messages", 
                            placeholder="💬 Ask me anything about your uploaded documents!",
                            height=500
                        )
                        gr.ChatInterface(
                            fn=self.chat_with_tutor, 
                            chatbot=chatbot,
                            title=""
                        )
                    
                    # Quiz Tab - UPDATED
                    with gr.Tab("📝 Quiz"):
                        gr.Markdown("### 🎯 Generate and Take Quizzes")
                        
                        with gr.Row():
                            with gr.Column():
                                # Upfront subject and topic specification
                                with gr.Row():
                                    quiz_subject = gr.Textbox(label="Subject", placeholder="e.g., Data Structures, Mathematics, Physics")
                                    quiz_topic = gr.Textbox(label="Topic", placeholder="e.g., Binary Search, Calculus, Thermodynamics")
                                
                                with gr.Row():
                                    quiz_marks = gr.Number(label="Total Marks", value=10, minimum=1)
                                    quiz_type = gr.Dropdown(
                                        label="Quiz Type",
                                        choices=["MCQ", "Short Answer", "Fill-in-blank", "Mixed"],
                                        value="Mixed"
                                    )
                                

                                
                                generate_quiz_btn = gr.Button("🎯 Generate Quiz", variant="primary")
                                quiz_status = gr.Textbox(label="Status", interactive=False)
                        
                        # Quiz display area
                        quiz_display = gr.HTML(visible=False)
                        
                        # Quiz submission area - SIMPLIFIED
                        with gr.Column(visible=True):
                            gr.Markdown("### ✍️ Submit Your Answers")
                            
                            # Dynamic answer inputs (simplified)
                            answer_inputs = []
                            for i in range(10):  # Max 10 questions
                                answer_inputs.append(gr.Textbox(
                                    label=f"Answer {i+1}",
                                    visible=False,
                                    placeholder="Enter your answer"
                                ))
                        
                            submit_quiz_btn = gr.Button("📊 Submit Quiz", variant="primary")
                        
                        quiz_results = gr.Textbox(label="Quiz Results", lines=15, interactive=False)

                    # Reports Tab
                    with gr.Tab("📊 Reports"):
                        gr.Markdown("### 📈 Your Learning Progress Reports")
                        
                        with gr.Row():
                            subjects_dropdown = gr.Dropdown(
                                label="Select Subject",
                                choices=[],
                                allow_custom_value=True
                            )
                            refresh_subjects_btn = gr.Button("🔄 Refresh Subjects")
                            generate_report_btn = gr.Button("📊 Generate Report", variant="primary")
                        
                        subject_report = gr.Textbox(
                            label="Subject Report",
                            lines=20,
                            interactive=False,
                            placeholder="Select a subject and click 'Generate Report' to see your progress"
                        )
                    
                    # Documents Tab  
                    with gr.Tab("📚 Documents"):
                        gr.Markdown("### 📁 Manage Your Knowledge Base")
                        
                        # KB Status
                        kb_status = gr.Textbox(
                            label="Knowledge Base Status", 
                            interactive=False, 
                            lines=4,
                            value="Please login to access your personal knowledge base"
                        )
                        refresh_kb_btn = gr.Button("🔄 Refresh Status")
                        
                        # File Upload
                        upload_files_btn = gr.UploadButton(
                            "➕ Upload Files", 
                            file_types=[".pdf", ".docx", ".pptx", ".xls", ".xlsx", ".txt", ".zip"],
                            file_count="multiple",
                            variant="primary"
                        )
                        
                        system_output = gr.Textbox(label="System Messages", interactive=False, lines=4)
                        
                        # Visualization
                        gr.Markdown("#### Vector Visualization")
                        vis_dropdown = gr.Dropdown(
                            choices=["2D Visualization", "3D Visualization"],
                            label="Visualization Type",
                            value="2D Visualization"
                        )
                        show_vis_btn = gr.Button("📈 Show Visualization")
                        vis_output = gr.Textbox(label="Visualization Status", interactive=False)
                        
                        # Management
                        gr.Markdown("#### 🗑️ Management")
                        clear_kb_btn = gr.Button("🗑️ Clear My Knowledge Base", variant="stop")

            # Event handlers
            login_btn.click(
                self.login_user,
                inputs=[login_username, login_password],
                outputs=[auth_message, auth_section, main_section]
            ).then(
                self.get_kb_status,
                outputs=kb_status
            ).then(
                self.refresh_subjects_dropdown,
                outputs=subjects_dropdown
            )
            
            register_btn.click(
                self.register_user,
                inputs=[reg_username, reg_password, reg_email],
                outputs=[auth_message, auth_section, main_section]
            ).then(
                self.get_kb_status,
                outputs=kb_status
            ).then(
                self.refresh_subjects_dropdown,
                outputs=subjects_dropdown
            )
            
            logout_btn.click(
                self.logout_user,
                outputs=[auth_message, auth_section, main_section]
            )
            
            # Updated Quiz event handlers
            generate_quiz_btn.click(
                self.generate_quiz,
                inputs=[quiz_subject, quiz_topic, quiz_marks, quiz_type],
                outputs=[quiz_status, quiz_display]
            )
            
            submit_quiz_btn.click(
                self.submit_quiz_answers,
                inputs=answer_inputs,  # No more subject/topic inputs needed!
                outputs=quiz_results
            )
            
            # Report event handlers
            refresh_subjects_btn.click(
                self.refresh_subjects_dropdown,
                outputs=subjects_dropdown
            )
            
            generate_report_btn.click(
                self.get_subject_report,
                inputs=subjects_dropdown,
                outputs=subject_report
            )
            
            # Document event handlers
            refresh_kb_btn.click(
                self.get_kb_status,
                outputs=kb_status
            )
            
            show_vis_btn.click(
                self.show_visualization,
                inputs=vis_dropdown,
                outputs=vis_output
            )

            upload_files_btn.upload(
                self.add_files_to_kb,
                inputs=upload_files_btn,
                outputs=system_output
            ).then(
                self.get_kb_status,
                outputs=kb_status
            )
            
            clear_kb_btn.click(
                self.clear_knowledge_base,
                outputs=system_output
            ).then(
                self.get_kb_status,
                outputs=kb_status
            )
        
        return demo

    def launch(self):
        import os
    
        port = int(os.getenv("PORT", 7860))  # Railway gives a PORT env var
    
        print("🚀 Starting AI Tutor App with Quiz-Based Learning...")
        print(f"📝 FastAPI backend should be running on http://0.0.0.0:{port}")
        print("🎯 Focus: Generate quizzes and track learning through quiz reports")
    
        self.demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            debug=True
        )


    def __del__(self):
        """Cleanup on app deletion"""
        if self.core:
            try:
                self.core.close()
            except Exception as e:
                print(f"[WARNING] Error during cleanup: {e}")
