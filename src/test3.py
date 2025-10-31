import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import re
import os

class ComprehensiveDataAnalyst:
    def __init__(self, df, llm):
        self.df = df
        self.llm = llm
        self.pandas_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="openai-functions"
        )
        
        # Business context knowledge base
        self.business_context = {
            "company_info": "Sales analysis for e-commerce company",
            "data_period": "Current fiscal year data",
            "business_goals": ["Increase revenue", "Improve customer retention", "Optimize product mix"],
            "kpis": {"revenue": "total", "customers": "customer_id", "products": "product", "regions": "region"}
        }
    
    def classify_question_type(self, question):
        """Classify questions into handling categories"""
        question_lower = question.lower()
        
        # Data calculation questions (pandas agent)
        data_keywords = ['total', 'sum', 'count', 'average', 'mean', 'max', 'min', 'highest', 'lowest', 
                        'show', 'list', 'how many', 'what is', 'calculate', 'breakdown', 'by region',
                        'by product', 'top', 'bottom', 'best', 'worst', 'revenue', 'sales']
        
        # Business insight questions (interpretation needed)
        business_keywords = ['why', 'reason', 'cause', 'explain', 'interpret', 'meaning', 'insight',
                           'story', 'implication', 'context', 'background', 'trend explanation']
        
        # Recommendation questions (prescriptive)
        recommendation_keywords = ['should', 'recommend', 'suggest', 'advice', 'strategy', 'improve',
                                 'optimize', 'action', 'decision', 'what to do', 'how to']
        
        # Prediction questions
        prediction_keywords = ['predict', 'forecast', 'future', 'next', 'will', 'expect', 'projection',
                              'trend', 'growth', 'decline']
        
        # Meta questions (about data itself)
        meta_keywords = ['columns', 'structure', 'data types', 'missing', 'null', 'quality', 'sample',
                        'dataset', 'rows', 'format']
        
        if any(keyword in question_lower for keyword in data_keywords):
            return "data_calculation"
        elif any(keyword in question_lower for keyword in business_keywords):
            return "business_insight"
        elif any(keyword in question_lower for keyword in recommendation_keywords):
            return "recommendation"
        elif any(keyword in question_lower for keyword in prediction_keywords):
            return "prediction"
        elif any(keyword in question_lower for keyword in meta_keywords):
            return "meta_data"
        else:
            return "general"
    
    def handle_data_calculation(self, question):
        """Handle data calculation questions with pandas agent"""
        try:
            # Enhanced prompt for complete dataset analysis
            enhanced_question = f"""
            IMPORTANT: Use the ENTIRE dataset (all {len(self.df)} rows) for this analysis.
            
            Question: {question}
            
            Provide analysis using the complete DataFrame. Include relevant breakdowns and context.
            """
            
            response = self.pandas_agent.invoke({"input": enhanced_question})
            return response.get('output', str(response)) if isinstance(response, dict) else str(response)
        except Exception as e:
            return self.manual_calculation_fallback(question)
    
    def handle_business_insight(self, question):
        """Handle business interpretation questions"""
        # First get relevant data
        data_context = self.get_data_summary()
        
        prompt = f"""
        Based on this sales data analysis, provide business insights:
        
        Data Context: {data_context}
        
        Question: {question}
        
        Provide:
        1. Data-driven insights
        2. Possible business explanations
        3. Context and implications
        4. Key patterns or trends
        
        Format as:
        **QUESTION CLASSIFICATION:**
        - Type: Diagnostic
        - Reasoning: [explanation]
        
        **BUSINESS INSIGHTS:**
        [detailed insights]
        
        **IMPLICATIONS:**
        [what this means for business]
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def handle_recommendations(self, question):
        """Handle recommendation and strategy questions"""
        data_context = self.get_data_summary()
        
        prompt = f"""
        Based on this sales data, provide actionable recommendations:
        
        Data Context: {data_context}
        Business Goals: {self.business_context['business_goals']}
        
        Question: {question}
        
        Provide:
        1. Specific recommendations
        2. Priority ranking
        3. Expected impact
        4. Implementation steps
        
        Format as:
        **QUESTION CLASSIFICATION:**
        - Type: Prescriptive
        - Reasoning: [explanation]
        
        **RECOMMENDATIONS:**
        [ranked list of actions]
        
        **IMPLEMENTATION:**
        [how to execute]
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def handle_predictions(self, question):
        """Handle forecasting and prediction questions"""
        try:
            # Get trend data first
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                trend_data = self.df.groupby(self.df['date'].dt.to_period('M'))['total'].sum()
                trend_summary = str(trend_data.tail(6))  # Last 6 months
            else:
                trend_summary = "No date column available for trend analysis"
            
            prompt = f"""
            Based on this sales trend data, provide predictions:
            
            Recent Trend Data: {trend_summary}
            Question: {question}
            
            Provide:
            1. Trend analysis
            2. Prediction with rationale  
            3. Confidence level
            4. Factors that could affect prediction
            
            Format as:
            **QUESTION CLASSIFICATION:**
            - Type: Predictive
            - Reasoning: [explanation]
            
            **TREND ANALYSIS:**
            [current patterns]
            
            **PREDICTIONS:**
            [forecasts with reasoning]
            
            **CONFIDENCE & RISKS:**
            [reliability and factors]
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Prediction analysis failed: {str(e)}. Need more structured time-series data for accurate forecasting."
    
    def handle_meta_data(self, question):
        """Handle questions about the dataset structure"""
        info = {
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "data_types": dict(self.df.dtypes.astype(str)),
            "missing_values": dict(self.df.isnull().sum()),
            "sample": self.df.head(3).to_dict(),
            "date_range": self.get_date_range() if 'date' in self.df.columns else "No date column"
        }
        
        return f"""
**QUESTION CLASSIFICATION:**
- Type: Descriptive (Meta-data)
- Reasoning: Question about dataset structure and properties

**DATASET INFORMATION:**
• Rows: {info['rows']:,}
• Columns: {info['columns']}
• Data Types: {info['data_types']}
• Missing Values: {info['missing_values']}
• Date Range: {info['date_range']}

**SAMPLE DATA:**
{pd.DataFrame(info['sample']).to_string()}
        """
    
    def handle_general(self, question):
        """Handle general questions that don't fit other categories"""
        prompt = f"""
        This question doesn't clearly fit standard data analysis categories.
        
        Question: {question}
        Available Data: Sales data with columns {list(self.df.columns)}
        
        Please:
        1. Clarify what specific analysis would be helpful
        2. Suggest alternative questions that could provide insights
        3. Provide any relevant general guidance
        
        Format your response with the analytics framework classification.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_data_summary(self):
        """Get key data summary for context"""
        summary = {
            "total_revenue": self.df['total'].sum(),
            "total_orders": len(self.df),
            "avg_order_value": self.df['total'].mean(),
            "top_regions": self.df.groupby('region')['total'].sum().sort_values(ascending=False).head(3).to_dict(),
            "top_products": self.df.groupby('product')['total'].sum().sort_values(ascending=False).head(3).to_dict(),
            "date_range": self.get_date_range() if 'date' in self.df.columns else "Unknown"
        }
        return summary
    
    def get_date_range(self):
        """Get date range from data"""
        try:
            dates = pd.to_datetime(self.df['date'], errors='coerce')
            return f"{dates.min()} to {dates.max()}"
        except:
            return "Date parsing failed"
    
    def manual_calculation_fallback(self, question):
        """Manual calculation when agent fails"""
        question_lower = question.lower()
        
        try:
            if 'total revenue' in question_lower or 'revenue' in question_lower:
                total = self.df['total'].sum()
                count = len(self.df)
                avg = self.df['total'].mean()
                return f"""
**MANUAL CALCULATION:**
• Total Revenue: ${total:,.2f}
• Total Orders: {count:,}  
• Average Order Value: ${avg:.2f}
• Top Region: {self.df.groupby('region')['total'].sum().idxmax()}
                """
            elif 'product' in question_lower:
                top_products = self.df.groupby('product')['total'].sum().sort_values(ascending=False).head(5)
                return f"**TOP PRODUCTS BY REVENUE:**\n{top_products.to_string()}"
            else:
                return "Manual calculation not available for this question type."
        except Exception as e:
            return f"Manual calculation failed: {str(e)}"
    
    def analyze_question(self, question):
        """Main method to handle any question"""
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING: {question}")
        print(f"{'='*60}")
        
        # Classify the question
        question_type = self.classify_question_type(question)
        print(f"📊 Question Type: {question_type}")
        
        # Route to appropriate handler
        if question_type == "data_calculation":
            result = self.handle_data_calculation(question)
        elif question_type == "business_insight":
            result = self.handle_business_insight(question)
        elif question_type == "recommendation":
            result = self.handle_recommendations(question)
        elif question_type == "prediction":
            result = self.handle_predictions(question)
        elif question_type == "meta_data":
            result = self.handle_meta_data(question)
        else:
            result = self.handle_general(question)
        
        print(f"\n{'='*60}")
        print("📋 COMPREHENSIVE ANALYSIS RESULT:")
        print(f"{'='*60}")
        print(result)
        print(f"{'='*60}")
        
        return result

# Usage
def main():
    # Setup (using your existing configuration)
    CUSTOM_BASE_URL = "https://llmmux.channels-ai.online/v1"
    API_KEY = "VSaNFvQ6eUWHw4y3oLrNDUnOiFbzEyUJfKhAAYeFwCuAr8X3BBCMyAlEMLclfFni"
    
    os.environ["OPENAI_API_KEY"] = API_KEY
    
    # Load data
    df = pd.read_csv("data.csv")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-oss-120b",
        temperature=0,
        api_key=API_KEY,
        base_url=CUSTOM_BASE_URL,
    )
    
    # Create comprehensive analyst
    analyst = ComprehensiveDataAnalyst(df, llm)
    
    # Interactive session
    print(f"\n{'='*60}")
    print("🚀 COMPREHENSIVE DATA ANALYST - 100% QUESTION COVERAGE")
    print(f"{'='*60}")
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\n🎯 Can handle ALL question types:")
    print("• Data calculations (What's our revenue?)")
    print("• Business insights (Why did sales drop?)")  
    print("• Recommendations (What should we do?)")
    print("• Predictions (What will happen next?)")
    print("• Meta questions (What data do we have?)")
    print("• General questions (Any other inquiry)")
    
    while True:
        question = input(f"\n❓ Your question (or 'quit'): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using the Comprehensive Data Analyst!")
            break
            
        if not question:
            continue
            
        analyst.analyze_question(question)

if __name__ == "__main__":
    main()