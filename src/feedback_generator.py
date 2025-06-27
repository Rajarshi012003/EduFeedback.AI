import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

class FeedbackGenerator:
    def __init__(self, ml_model_path, llm_model_path):
        print("Loading ML model...")
        with open(ml_model_path, 'rb') as f:
            self.ml_model = pickle.load(f)
        
        # Get the exact feature names the model was trained on
        if hasattr(self.ml_model, 'feature_names_in_'):
            self.expected_features = list(self.ml_model.feature_names_in_)
        else:
            if hasattr(self.ml_model, 'named_steps'):
                scaler = self.ml_model.named_steps.get('scaler')
                if hasattr(scaler, 'feature_names_in_'):
                    self.expected_features = list(scaler.feature_names_in_)
                else:
                    self.expected_features = self.get_features_from_dataset()
            else:
                self.expected_features = self.get_features_from_dataset()
        
        print(f"Model expects {len(self.expected_features)} features")
        print(f"First 10 features: {self.expected_features[:10]}")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Loading LLM with memory optimization...")
        
        # Create offload directory
        offload_dir = "./model_offload"
        os.makedirs(offload_dir, exist_ok=True)
        
        try:
            # Configure quantization with CPU offloading enabled
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )
            
            # Custom device map for your GPU constraints
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": "cpu",
                "model.layers.11": "cpu",
                "model.layers.12": "cpu",
                "model.layers.13": "cpu",
                "model.layers.14": "cpu",
                "model.layers.15": "cpu",
                "model.layers.16": "cpu",
                "model.layers.17": "cpu",
                "model.layers.18": "cpu",
                "model.layers.19": "cpu",
                "model.layers.20": "cpu",
                "model.layers.21": "cpu",
                "model.layers.22": "cpu",
                "model.layers.23": "cpu",
                "model.layers.24": "cpu",
                "model.layers.25": "cpu",
                "model.layers.26": "cpu",
                "model.layers.27": "cpu",
                "model.layers.28": "cpu",
                "model.layers.29": "cpu",
                "model.layers.30": "cpu",
                "model.layers.31": "cpu",
                "model.norm": "cpu",
                "lm_head": "cpu"
            }
            
            # Load base model with custom device mapping
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                quantization_config=bnb_config,
                device_map=device_map,
                offload_folder=offload_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load PEFT adapter
            try:
                self.model = PeftModel.from_pretrained(
                    base_model, 
                    llm_model_path,
                    is_trainable=False,
                    offload_folder=offload_dir
                )
                print("✅ Fine-tuned PEFT adapter loaded successfully!")
            except Exception as e:
                print(f"❌ PEFT loading failed: {e}")
                print("Using base model without fine-tuning...")
                self.model = base_model
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ GPU loading failed: {e}")
            print("Falling back to CPU-only mode...")
            self.load_cpu_only_model(llm_model_path)
    
    def load_cpu_only_model(self, llm_model_path):
        """Fallback to CPU-only model loading"""
        torch.cuda.empty_cache()
        
        # Load model entirely on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try PEFT on CPU
        try:
            self.model = PeftModel.from_pretrained(base_model, llm_model_path, is_trainable=False)
            print("✅ PEFT adapter loaded on CPU!")
        except:
            print("Using base model on CPU...")
            self.model = base_model
    
    def get_features_from_dataset(self):
        """Get feature names from the processed dataset"""
        try:
            df = pd.read_csv("data/processed/student_processed.csv")
            exclude_cols = ['G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y', 
                          'avg_grade_x', 'avg_grade_y', 'overall_avg_grade']
            features = [col for col in df.columns if col not in exclude_cols]
            return features
        except:
            return [
                'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
                'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
            ]
    
    def prepare_features_from_sample(self, student_data):
        """Convert sample student data to full feature set"""
        features = {}
        
        for feature in self.expected_features:
            if feature in student_data:
                features[feature] = student_data[feature]
            elif feature == 'studytime':
                features[feature] = student_data.get('studytime', 2)
            elif feature == 'health':
                features[feature] = student_data.get('health', 3)
            elif feature == 'famrel':
                features[feature] = student_data.get('family_support', 3)
            elif feature == 'age':
                features[feature] = 16
            elif feature in ['Medu', 'Fedu']:
                features[feature] = 3
            elif feature in ['school', 'sex', 'address', 'famsize', 'Pstatus']:
                features[feature] = 1
            elif feature in ['failures', 'absences']:
                features[feature] = 0
            elif feature in ['freetime', 'goout', 'Dalc', 'Walc']:
                features[feature] = 2
            else:
                features[feature] = 1
        
        return features
    
    def predict_performance(self, student_data):
        """Predict student performance using ML model"""
        prepared_features = self.prepare_features_from_sample(student_data)
        df = pd.DataFrame([prepared_features])
        df = df[self.expected_features]
        df = df.fillna(0.0)
        
        prediction = self.ml_model.predict(df)[0]
        
        if prediction >= 16:
            return "Excellent", prediction
        elif prediction >= 14:
            return "Good", prediction
        elif prediction >= 12:
            return "Average", prediction
        elif prediction >= 10:
            return "Below Average", prediction
        else:
            return "Needs Improvement", prediction
    
    def identify_weak_areas(self, student_data):
        """Identify areas needing improvement"""
        weak_areas = []
        
        if student_data.get('studytime', 3) < 2:
            weak_areas.append("Study Time Management")
        if student_data.get('health', 3) < 2:
            weak_areas.append("Health and Wellness")
        if student_data.get('family_support', 3) < 2:
            weak_areas.append("Family Support Utilization")
        
        return weak_areas
    
    def construct_prompt(self, student_data, performance, prediction_score, weak_areas):
        """Create detailed prompt for LLM"""
        prompt = f"### Instruction:\nGenerate a detailed and comprehensive personalized educational feedback for a Portuguese student with {performance} performance (predicted grade: {prediction_score:.1f}/20).\n\n"
        
        # Add more context
        prompt += f"Student Profile:\n"
        prompt += f"- Current Performance Level: {performance}\n"
        prompt += f"- Predicted Grade: {prediction_score:.1f} out of 20\n"
        
        if weak_areas:
            weak_str = ", ".join(weak_areas)
            prompt += f"- Areas needing improvement: {weak_str}\n"
        else:
            prompt += f"- Overall performance is satisfactory\n"
        
        prompt += f"\nPlease provide:\n"
        prompt += f"1. Specific strengths and positive aspects\n"
        prompt += f"2. Areas for improvement with concrete suggestions\n"
        prompt += f"3. Actionable study strategies and recommendations\n"
        prompt += f"4. Motivational and encouraging advice\n"
        prompt += f"5. Next steps for academic progress\n\n"
        prompt += f"Use a supportive, encouraging, and detailed tone. Aim for comprehensive feedback.\n\n### Response:\n"
        
        return prompt

    
    def generate_feedback(self, prompt):
        """Generate detailed feedback using LLM"""
        print("🔄 Generating detailed feedback...")
        
        # Longer prompt for more context
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
        
        # Find device of the first model parameter
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,        # Increased from 30 to 200
                min_length=100,            # Ensure minimum length
                temperature=0.8,           # Increased for more creativity
                do_sample=True,            # Enable sampling for variety
                top_p=0.95,               # Nucleus sampling for quality
                repetition_penalty=1.2,    # Prevent repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3     # Prevent 3-gram repetition
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback = full_response.replace(prompt, "").strip()
        
        # Ensure we have meaningful feedback
        if len(feedback) < 50:
            feedback = f"Based on your {performance.lower()} performance (Grade: {prediction_score:.1f}/20), here are some recommendations: Focus on consistent study habits, seek help when needed, practice regularly, and maintain a positive attitude towards learning. Your current grade shows room for improvement, so consider reviewing fundamental concepts and asking teachers for additional support."
        
        return feedback

    
    def generate(self, student_data):
        """Full feedback generation pipeline"""
        try:
            performance, prediction_score = self.predict_performance(student_data)
            weak_areas = self.identify_weak_areas(student_data)
            prompt = self.construct_prompt(student_data, performance, prediction_score, weak_areas)
            feedback = self.generate_feedback(prompt)
            
            return {
                "performance": performance,
                "prediction_score": float(prediction_score),
                "weak_areas": weak_areas,
                "feedback": feedback,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

def test_generator():
    """Test the feedback generator with sample data"""
    import warnings
    warnings.filterwarnings("ignore")
    
    # Set environment variable for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    sample_student = {
        'studytime': 2,
        'health': 3,
        'family_support': 4
    }
    
    generator = FeedbackGenerator(
        ml_model_path=r"models/predictive/best_model.pkl",
        llm_model_path=r"models/llm/fine_tuned_model"
    )
    
    result = generator.generate(sample_student)
    
    if result['status'] == 'success':
        print("Generated Feedback:")
        print("-" * 50)
        print(f"Performance: {result['performance']} (Grade: {result['prediction_score']:.1f}/20)")
        print(f"Weak Areas: {result['weak_areas']}")
        print(f"\nFeedback:\n{result['feedback']}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_generator()