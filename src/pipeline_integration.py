import pandas as pd
import json
import numpy as np
from feedback_generator import FeedbackGenerator
from datetime import datetime
import os

class StudentFeedbackSystem:
    def __init__(self, data_path, ml_model_path, llm_model_path):
        print("Initializing Student Feedback System...")
        self.data = pd.read_csv(data_path)
        self.generator = FeedbackGenerator(ml_model_path, llm_model_path)
        print(f"Loaded data for {len(self.data)} students")
        
    def get_student_data(self, student_id):
        """Retrieve student data by ID"""
        # For the Portuguese dataset, look for any ID column
        id_columns = ['user_id', 'student_id', 'id']
        id_col = None
        
        for col in id_columns:
            if col in self.data.columns:
                id_col = col
                break
        
        if id_col is None:
            # Use index if no ID column found
            if student_id < len(self.data):
                student_dict = self.data.iloc[student_id].to_dict()
                return student_dict
            return None
        
        student = self.data[self.data[id_col] == student_id]
        if student.empty:
            return None
        
        # Convert to dictionary and remove ID columns
        student_dict = student.iloc[0].to_dict()
        exclude_cols = [id_col, 'G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y']
        for col in exclude_cols:
            student_dict.pop(col, None)
        
        return student_dict
    
    def generate_feedback(self, student_id):
        """Generate feedback for a student"""
        student_data = self.get_student_data(student_id)
        if not student_data:
            return {"error": "Student not found", "student_id": student_id}
        
        feedback = self.generator.generate(student_data)
        feedback["student_id"] = student_id
        feedback["timestamp"] = datetime.now().isoformat()
        return feedback
    
    def generate_batch_feedback(self, num_students=3):  # Reduced from 5 to 3
        """Generate feedback for multiple students"""
        print(f"Generating feedback for {num_students} students...")
        
        # Use first few students instead of random for consistency
        available_students = min(num_students, len(self.data))
        sample_students = list(range(available_students))
        
        batch_results = {}
        successful_count = 0
        
        for i, student_id in enumerate(sample_students):
            print(f"Processing student {i+1}/{len(sample_students)} (ID: {student_id})...")
            feedback = self.generate_feedback(student_id)
            batch_results[str(student_id)] = feedback
            
            if feedback.get('status') == 'success':
                successful_count += 1
                print(f"✅ Success - Performance: {feedback.get('performance', 'N/A')}")
            else:
                print(f"❌ Failed - Error: {feedback.get('error', 'Unknown')}")
        
        # Add batch summary
        batch_results['_batch_summary'] = {
            'total_students': len(sample_students),
            'successful_feedback': successful_count,
            'failed_feedback': len(sample_students) - successful_count,
            'success_rate': successful_count / len(sample_students) if sample_students else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return batch_results
    
    def save_feedback(self, feedback, path):
        """Save feedback to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(feedback, f, indent=2, default=str)
        print(f"Feedback saved to {path}")

def main():
    # Initialize system with Portuguese student dataset
    system = StudentFeedbackSystem(
        data_path=r"data/processed/student_processed.csv",  # Updated path
        ml_model_path=r"models/predictive/best_model.pkl",
        llm_model_path=r"models/llm/fine_tuned_model"
    )
    
    # Create output directory
    os.makedirs(r"outputs", exist_ok=True)
    
    # Generate feedback for a single student
    sample_student_id = 0  # Use first student
    print(f"\n=== Generating Single Student Feedback ===")
    single_feedback = system.generate_feedback(sample_student_id)
    system.save_feedback(single_feedback, r"outputs/single_student_feedback.json")
    
    # Generate batch feedback (reduced to 3 students)
    print(f"\n=== Generating Batch Feedback ===")
    batch_feedback = system.generate_batch_feedback(num_students=3)
    system.save_feedback(batch_feedback, r"outputs/batch_feedback.json")
    
    # Print results summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Single student feedback generated for student {sample_student_id}")
    if single_feedback.get('status') == 'success':
        print(f"Performance: {single_feedback['performance']}")
        print(f"Prediction Score: {single_feedback['prediction_score']:.1f}/20")
        print(f"Sample Feedback: {single_feedback['feedback'][:100]}...")
    
    batch_summary = batch_feedback.get('_batch_summary', {})
    print(f"\nBatch feedback results:")
    print(f"Success Rate: {batch_summary.get('successful_feedback', 0)}/{batch_summary.get('total_students', 0)}")
    print(f"Success Percentage: {batch_summary.get('success_rate', 0)*100:.1f}%")
    print("All results saved to outputs/")

if __name__ == "__main__":
    main()
