import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os
import pickle
from datetime import datetime

# Try to import NLTK and ROUGE, but handle if not available
try:
    from nltk.translate.bleu_score import sentence_bleu
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. BLEU scores will be skipped.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("ROUGE not available. ROUGE scores will be skipped.")

class Evaluator:
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def evaluate_ml_model(self, model_path, test_data_path):
        """Evaluate ML model performance"""
        print("Evaluating ML model performance...")
        
        try:
            # Load model and test data
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            test_data = pd.read_csv(test_data_path)
            
            # Find target column for Portuguese dataset
            target_cols = ['G3_y', 'G3_x', 'correct_mean', 'engagement_score', 'performance_score']
            target_col = None
            for col in target_cols:
                if col in test_data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                return {"error": "No suitable target column found"}
            
            print(f"Using target column: {target_col}")
            
            # Prepare features and target
            exclude_cols = [
                target_col, 'G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y',
                'avg_grade_x', 'avg_grade_y', 'overall_avg_grade',
                'user_id', 'student_id', 'id'
            ]
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]
            
            X = test_data[feature_cols].fillna(test_data[feature_cols].mean())
            y = test_data[target_col]
            
            print(f"Features: {len(feature_cols)}, Samples: {len(y)}")
            
            # Predict and calculate metrics
            y_pred = model.predict(X)
            
            return {
                "target_column": target_col,
                "num_samples": len(y),
                "num_features": len(feature_cols),
                "mse": float(mean_squared_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(mean_absolute_error(y, y_pred)),
                "r2": float(r2_score(y, y_pred)),
                "mean_actual": float(np.mean(y)),
                "mean_predicted": float(np.mean(y_pred)),
                "prediction_range": {
                    "min": float(np.min(y_pred)),
                    "max": float(np.max(y_pred))
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_feedback_quality(self, feedback_path):
        """Evaluate feedback quality using automated metrics"""
        print("Evaluating feedback quality...")
        
        try:
            with open(feedback_path, 'r') as f:
                feedback_data = json.load(f)
            
            # Handle both single feedback and batch feedback
            if isinstance(feedback_data, dict) and 'feedback' in feedback_data:
                # Single feedback
                feedbacks = [feedback_data]
            else:
                # Batch feedback - exclude metadata entries
                feedbacks = [v for k, v in feedback_data.items() 
                           if not k.startswith('_') and isinstance(v, dict) and 'feedback' in v]
            
            if not feedbacks:
                return {"error": "No valid feedback found"}
            
            # Analyze feedback content
            feedback_texts = [f['feedback'] for f in feedbacks if f.get('status') == 'success']
            
            if not feedback_texts:
                return {"error": "No successful feedback generations found"}
            
            print(f"Analyzing {len(feedback_texts)} feedback texts...")
            
            # Calculate text metrics
            word_counts = [len(text.split()) for text in feedback_texts]
            sentence_counts = [text.count('.') + text.count('!') + text.count('?') for text in feedback_texts]
            
            # Content analysis - updated for educational context
            positive_words = ['good', 'great', 'excellent', 'well', 'improve', 'progress', 'strength', 'potential', 'capable']
            actionable_words = ['try', 'practice', 'focus', 'consider', 'work', 'review', 'study', 'develop', 'build']
            encouraging_words = ['keep', 'continue', 'maintain', 'develop', 'can', 'will', 'able', 'succeed']
            educational_words = ['learn', 'understand', 'concept', 'skill', 'knowledge', 'grade', 'performance']
            
            positive_count = sum(1 for text in feedback_texts 
                               if any(word in text.lower() for word in positive_words))
            actionable_count = sum(1 for text in feedback_texts 
                                 if any(word in text.lower() for word in actionable_words))
            encouraging_count = sum(1 for text in feedback_texts 
                                  if any(word in text.lower() for word in encouraging_words))
            educational_count = sum(1 for text in feedback_texts 
                                  if any(word in text.lower() for word in educational_words))
            
            # Performance distribution
            performance_levels = [f.get('performance', 'Unknown') for f in feedbacks if f.get('status') == 'success']
            performance_dist = {level: performance_levels.count(level) for level in set(performance_levels)}
            
            # Grade analysis for Portuguese system (0-20)
            prediction_scores = [f.get('prediction_score', 0) for f in feedbacks if f.get('status') == 'success']
            
            return {
                "total_feedback": len(feedbacks),
                "successful_feedback": len(feedback_texts),
                "avg_word_count": float(np.mean(word_counts)),
                "avg_sentence_count": float(np.mean(sentence_counts)),
                "word_count_range": {
                    "min": int(min(word_counts)),
                    "max": int(max(word_counts))
                },
                "content_quality": {
                    "positive_feedback_ratio": positive_count / len(feedback_texts),
                    "actionable_feedback_ratio": actionable_count / len(feedback_texts),
                    "encouraging_feedback_ratio": encouraging_count / len(feedback_texts),
                    "educational_feedback_ratio": educational_count / len(feedback_texts)
                },
                "performance_distribution": performance_dist,
                "grade_analysis": {
                    "avg_predicted_grade": float(np.mean(prediction_scores)) if prediction_scores else 0,
                    "grade_range": {
                        "min": float(min(prediction_scores)) if prediction_scores else 0,
                        "max": float(max(prediction_scores)) if prediction_scores else 0
                    }
                },
                "sample_feedback": feedback_texts[0] if feedback_texts else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_comprehensive_report(self, ml_metrics, feedback_metrics, output_path):
        """Generate comprehensive evaluation report"""
        
        # Calculate overall scores
        ml_score = max(0, ml_metrics.get('r2', 0)) if 'error' not in ml_metrics else 0
        
        if 'error' not in feedback_metrics:
            content_quality = feedback_metrics.get('content_quality', {})
            feedback_score = (
                content_quality.get('positive_feedback_ratio', 0) * 0.25 +
                content_quality.get('actionable_feedback_ratio', 0) * 0.35 +
                content_quality.get('encouraging_feedback_ratio', 0) * 0.25 +
                content_quality.get('educational_feedback_ratio', 0) * 0.15
            )
        else:
            feedback_score = 0
        
        overall_score = ml_score * 0.6 + feedback_score * 0.4
        
        report = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_type": "Portuguese Student Performance",
                "ml_model_evaluated": 'error' not in ml_metrics,
                "feedback_evaluated": 'error' not in feedback_metrics,
                "sample_size": feedback_metrics.get('total_feedback', 0) if 'error' not in feedback_metrics else 0
            },
            "ml_model_performance": ml_metrics,
            "feedback_quality": feedback_metrics,
            "overall_assessment": {
                "ml_component_score": float(ml_score),
                "feedback_component_score": float(feedback_score),
                "overall_system_score": float(overall_score),
                "system_rating": self.get_system_rating(overall_score),
                "recommendations": self.generate_recommendations(ml_metrics, feedback_metrics)
            }
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def get_system_rating(self, score):
        """Convert numerical score to rating"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Satisfactory"
        elif score >= 0.4:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def generate_recommendations(self, ml_metrics, feedback_metrics):
        """Generate recommendations for system improvement"""
        recommendations = []
        
        # ML model recommendations
        if 'error' not in ml_metrics:
            r2_score = ml_metrics.get('r2', 0)
            if r2_score < 0.7:
                recommendations.append("Consider feature engineering or trying different ML algorithms to improve prediction accuracy")
            if r2_score > 0.9:
                recommendations.append("Excellent ML performance - model generalizes well to Portuguese student data")
        
        # Feedback quality recommendations
        if 'error' not in feedback_metrics:
            content_quality = feedback_metrics.get('content_quality', {})
            avg_words = feedback_metrics.get('avg_word_count', 0)
            
            if content_quality.get('actionable_feedback_ratio', 0) < 0.8:
                recommendations.append("Improve LLM fine-tuning data to include more actionable educational advice")
            
            if avg_words < 50:
                recommendations.append("Increase feedback length for more comprehensive guidance")
            elif avg_words > 200:
                recommendations.append("Consider shortening feedback for better readability")
            
            if content_quality.get('educational_feedback_ratio', 0) < 0.7:
                recommendations.append("Enhance educational terminology and context in feedback generation")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory for Portuguese student feedback generation")
        
        return recommendations
    
    def print_evaluation_summary(self, report):
        """Print formatted evaluation summary"""
        print("\n" + "="*70)
        print("🎓 AUTOMATED STUDENT FEEDBACK SYSTEM - EVALUATION REPORT")
        print("="*70)
        
        metadata = report.get('evaluation_metadata', {})
        overall = report['overall_assessment']
        
        print(f"\n📊 DATASET: {metadata.get('dataset_type', 'Unknown')}")
        print(f"📅 EVALUATION DATE: {metadata.get('timestamp', 'Unknown')}")
        print(f"📈 SAMPLE SIZE: {metadata.get('sample_size', 'Unknown')} students")
        
        print(f"\n🎯 OVERALL SYSTEM SCORE: {overall['overall_system_score']:.3f}/1.000")
        print(f"⭐ SYSTEM RATING: {overall['system_rating']}")
        
        # ML Performance
        ml_perf = report['ml_model_performance']
        if 'error' not in ml_perf:
            print(f"\n🤖 ML MODEL PERFORMANCE:")
            print(f"   • Target Variable: {ml_perf['target_column']}")
            print(f"   • R² Score: {ml_perf['r2']:.4f}")
            print(f"   • RMSE: {ml_perf['rmse']:.4f}")
            print(f"   • Features Used: {ml_perf['num_features']}")
            print(f"   • Grade Range: {ml_perf['prediction_range']['min']:.1f} - {ml_perf['prediction_range']['max']:.1f}")
        else:
            print(f"\n❌ ML MODEL EVALUATION FAILED: {ml_perf['error']}")
        
        # Feedback Quality
        fb_qual = report['feedback_quality']
        if 'error' not in fb_qual:
            print(f"\n💬 FEEDBACK QUALITY:")
            print(f"   • Total Generated: {fb_qual['total_feedback']}")
            print(f"   • Success Rate: {fb_qual['successful_feedback']}/{fb_qual['total_feedback']}")
            print(f"   • Avg Length: {fb_qual['avg_word_count']:.1f} words")
            print(f"   • Word Range: {fb_qual['word_count_range']['min']}-{fb_qual['word_count_range']['max']} words")
            
            content_quality = fb_qual.get('content_quality', {})
            print(f"   • Positive Tone: {content_quality.get('positive_feedback_ratio', 0):.2%}")
            print(f"   • Actionable Content: {content_quality.get('actionable_feedback_ratio', 0):.2%}")
            print(f"   • Encouraging Tone: {content_quality.get('encouraging_feedback_ratio', 0):.2%}")
            print(f"   • Educational Context: {content_quality.get('educational_feedback_ratio', 0):.2%}")
            
            grade_analysis = fb_qual.get('grade_analysis', {})
            print(f"   • Avg Predicted Grade: {grade_analysis.get('avg_predicted_grade', 0):.1f}/20")
            
            if fb_qual.get('sample_feedback'):
                print(f"\n📝 SAMPLE FEEDBACK:")
                print(f"   {fb_qual['sample_feedback'][:200]}...")
        else:
            print(f"\n❌ FEEDBACK EVALUATION FAILED: {fb_qual['error']}")
        
        # Recommendations
        recommendations = overall.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("="*70)

def main():
    """Main evaluation function"""
    evaluator = Evaluator()
    
    # File paths for Portuguese student dataset
    ml_model_path = r"models/predictive/best_model.pkl"
    test_data_path = r"data/processed/student_processed.csv"
    feedback_path = r"outputs/batch_feedback.json"
    output_path = r"outputs/evaluation_report.json"
    
    # Check if files exist
    missing_files = []
    for path in [ml_model_path, test_data_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return
    
    # Evaluate ML model
    print("🔍 Starting ML model evaluation...")
    ml_metrics = evaluator.evaluate_ml_model(ml_model_path, test_data_path)
    
    # Evaluate feedback quality (if feedback file exists)
    if os.path.exists(feedback_path):
        print("🔍 Starting feedback quality evaluation...")
        feedback_metrics = evaluator.evaluate_feedback_quality(feedback_path)  # Fixed method name
    else:
        print(f"Feedback file not found: {feedback_path}")
        print("Run pipeline_integration.py first to generate feedback.")
        feedback_metrics = {"error": "No feedback file found"}
    
    # Generate comprehensive report
    print("📊 Generating comprehensive report...")
    report = evaluator.generate_comprehensive_report(
        ml_metrics, feedback_metrics, output_path
    )
    
    # Print summary
    evaluator.print_evaluation_summary(report)
    
    print(f"\n💾 Detailed report saved to: {output_path}")

if __name__ == "__main__":
    main()

