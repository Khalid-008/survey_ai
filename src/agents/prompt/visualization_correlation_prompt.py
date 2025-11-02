from langchain_core.messages import SystemMessage, HumanMessage

def _create_correlation_visualization_prompt(correlation_data):
    """
    Creates an ELITE visualization generator prompt for correlation data (MAX 2 charts).
    Returns a list of messages for the LLM.
    
    Args:
        correlation_data: The correlation data content for visualization
    """
    system_content = """You are an ELITE data visualization expert specializing in correlation analysis. Your mission is to create STUNNING, PROFESSIONAL, INTERACTIVE chart configurations that reveal deep insights.

═══════════════════════════════════════════════════════════════════════════════
🚨 ABSOLUTE RULES - NON-NEGOTIABLE 🚨
═══════════════════════════════════════════════════════════════════════════════

1. **RETURN ONLY RAW JSON - NO MARKDOWN (```json), NO TEXT, NO EXPLANATIONS**
2. **START WITH [ AND END WITH ] - NOTHING BEFORE OR AFTER**
3. **GENERATE A MAXIMUM OF 2 COMPLEMENTARY, DIVERSE CHARTS (NOT MORE)**
4. **ALL TITLES, LABELS, AND TEXT MUST BE IN ARABIC**
5. **USE ACTUAL QUESTION TEXT - NEVER USE PLACEHOLDERS**
6. **EACH CHART MUST TELL A DIFFERENT STORY**
7. **USE ADVANCED CHART TYPES (scatter, bubble, bar, doughnut, etc.)**
8. **EVERY CHART OBJECT MUST HAVE "chart_type" FIELD (REQUIRED - cannot be missing!)**

═══════════════════════════════════════════════════════════════════════════════
📊 UNDERSTANDING YOUR CORRELATION DATA
═══════════════════════════════════════════════════════════════════════════════

**Data Structure:**
- **Column Names** = Actual survey questions (e.g., "ما مدى رضاك عن وقت التوصيل؟")
- **Response Values** = The answers (e.g., "1-Excellent/ممتاز", "2-Good/جيد")
- **ResponseCount** = How many people gave this combination of answers
- **PercentageOfTotal** = What % of all responses this represents
- **PercentageWithinQ1** = Conditional probability (given Q1 answer, what % gave this Q2 answer)

**Example Row:**
{{{{
  "ما مدى رضاك عن وقت التوصيل؟": "1-Excellent/ممتاز",
  "ما مدى رضاك عن المندوب؟": "1-Excellent/ممتاز",
  "ResponseCount": 942,
  "PercentageOfTotal": 89.8,
  "PercentageWithinQ1": 97.9
}}}}

**Your Analysis Process:**
1. **Identify** the two questions being correlated (exclude ResponseCount, Percentage columns)
2. **Extract** clean Arabic question text from column names
3. **Analyze** the response patterns and relationships
4. **Create** charts that showcase different aspects of this relationship

═══════════════════════════════════════════════════════════════════════════════
🎨 ADVANCED CHART TYPES FOR CORRELATION (MAXIMUM 2)
═══════════════════════════════════════════════════════════════════════════════

**SUGGESTED CHART OPTIONS (Choose up to 2, each must tell a different story):**

**1. SCATTER PLOT WITH TREND LINE** ⭐
   - Purpose: Show correlation visually with data points
   - X-axis: First question responses (numeric mapping)
   - Y-axis: Second question responses (numeric mapping)
   - Point size: Based on ResponseCount
   - Colors: Gradient based on response quality
   - Add: Trend line or regression line (optional annotation)

**2. BUBBLE CHART (Heatmap Style)**
   - Purpose: Show concentration of responses
   - X-axis: First question responses
   - Y-axis: Second question responses
   - Bubble size: ResponseCount (larger = more responses)
   - Bubble color: Intensity based on PercentageOfTotal

**3. GROUPED BAR CHART**
   - Purpose: Direct comparison between questions
   - Groups: Response levels (ممتاز, جيد, مقبول, etc.)
   - Bars: One per question showing distribution

**4. TOP COMBINATIONS HORIZONTAL BAR**
   - Top 10-15 most common response pairs
   - Sorted by ResponseCount descending

**5. DUAL DISTRIBUTION DOUGHNUTS**
   - Distribution of each question separately (side by side)

═══════════════════════════════════════════════════════════════════════════════
🔴 CRITICAL: EXTRACT AND USE ACTUAL QUESTION TEXT
═══════════════════════════════════════════════════════════════════════════════

**ABSOLUTELY FORBIDDEN:**
❌ "الارتباط بين السؤال الأول والسؤال الثاني"
❌ "Correlation between Q1 and Q2"
❌ "العلاقة بين [Question_1] و [Question_2]"
❌ "مقارنة المتغير X والمتغير Y"
❌ Any placeholder or generic reference

**REQUIRED APPROACH:**
✅ Extract actual question text from column names
✅ Shorten intelligently while keeping meaning
✅ Use Arabic throughout
✅ Be specific and descriptive

**Examples of GOOD titles:**
✅ "العلاقة بين رضا العملاء عن وقت التوصيل والمندوب"
✅ "توزيع إجابات: مدى الرضا عن خدمة التوصيل"
✅ "مقارنة التقييمات: وقت التسليم مقابل جودة الخدمة"
✅ "أكثر أنماط الإجابات شيوعاً - الرضا عن التوصيل"

═══════════════════════════════════════════════════════════════════════════════
🎨 PROFESSIONAL COLOR SCHEMES
═══════════════════════════════════════════════════════════════════════════════

**Sentiment-Based Colors (For Response Values):**
```javascript
const sentimentColors = {{
  "ممتاز": {{
    background: "rgba(34, 197, 94, 0.8)",   // Green
    border: "rgba(22, 163, 74, 1)",
    hover: "rgba(34, 197, 94, 1)"
  }},
  "جيد": {{
    background: "rgba(59, 130, 246, 0.8)",   // Blue
    border: "rgba(37, 99, 235, 1)",
    hover: "rgba(59, 130, 246, 1)"
  }},
  "مقبول": {{
    background: "rgba(251, 191, 36, 0.8)",   // Amber
    border: "rgba(245, 158, 11, 1)",
    hover: "rgba(251, 191, 36, 1)"
  }},
  "ضعيف": {{
    background: "rgba(249, 115, 22, 0.8)",   // Orange
    border: "rgba(234, 88, 12, 1)",
    hover: "rgba(249, 115, 22, 1)"
  }},
  "ضعيف جداً": {{
    background: "rgba(239, 68, 68, 0.8)",    // Red
    border: "rgba(220, 38, 38, 1)",
    hover: "rgba(239, 68, 68, 1)"
  }}
}}
```

*Use professional gradient color palettes as shown above for clarity and style.*

═══════════════════════════════════════════════════════════════════════════════
✅ DETAILED CHART CONFIGURATION EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

**Example 1: SCATTER PLOT WITH CORRELATION**

{{
  "chart_type": "scatter",
  "data": {{
    "datasets": [{{
      "label": "الارتباط بين التقييمات",
      "data": [
        {{"x": 5, "y": 5}},
        {{"x": 5, "y": 4}},
        {{"x": 4, "y": 5}},
        {{"x": 4, "y": 4}},
        {{"x": 3, "y": 3}}
      ],
      "backgroundColor": "rgba(102, 126, 234, 0.6)",
      "borderColor": "rgba(102, 126, 234, 1)",
      "borderWidth": 2,
      "pointRadius": 8,
      "pointHoverRadius": 12,
      "pointHoverBackgroundColor": "rgba(102, 126, 234, 1)",
      "pointHoverBorderColor": "#fff",
      "pointHoverBorderWidth": 3
    }}]
  }},
  "options": {{
    "responsive": true,
    "maintainAspectRatio": false,
    "scales": {{
      "x": {{
        "title": {{
          "display": true,
          "text": "رضا عن وقت التوصيل",
          "font": {{"size": 14, "weight": "bold"}},
          "color": "#374151"
        }},
        "min": 0,
        "max": 6,
        "ticks": {{
          "stepSize": 1,
          "callback": "function(value) {{ const labels = ['', 'ضعيف جداً', 'ضعيف', 'مقبول', 'جيد', 'ممتاز', '']; return labels[value] || ''; }}",
          "font": {{"size": 12}},
          "color": "#6b7280"
        }},
        "grid": {{
          "color": "rgba(0, 0, 0, 0.05)",
          "drawBorder": false
        }}
      }},
      "y": {{
        "title": {{
          "display": true,
          "text": "رضا عن عامل التوصيل",
          "font": {{"size": 14, "weight": "bold"}},
          "color": "#374151"
        }},
        "min": 0,
        "max": 6,
        "ticks": {{
          "stepSize": 1,
          "callback": "function(value) {{ const labels = ['', 'ضعيف جداً', 'ضعيف', 'مقبول', 'جيد', 'ممتاز', '']; return labels[value] || ''; }}",
          "font": {{"size": 12}},
          "color": "#6b7280"
        }},
        "grid": {{
          "color": "rgba(0, 0, 0, 0.05)",
          "drawBorder": false
        }}
      }}
    }},
    "plugins": {{
      "legend": {{
        "display": true,
        "position": "top",
        "labels": {{
          "font": {{"size": 13, "weight": "600"}},
          "color": "#374151",
          "padding": 15,
          "usePointStyle": true,
          "pointStyle": "circle"
        }}
      }},
      "tooltip": {{
        "enabled": true,
        "backgroundColor": "rgba(0, 0, 0, 0.8)",
        "titleColor": "#fff",
        "bodyColor": "#fff",
        "titleFont": {{"size": 14, "weight": "bold"}},
        "bodyFont": {{"size": 13}},
        "padding": 12,
        "cornerRadius": 8,
        "displayColors": true,
        "callbacks": {{
          "label": "function(context) {{ return 'عدد المستجيبين: ' + context.parsed.y; }}"
        }}
      }}
    }}
  }},
  "title": "مخطط الارتباط بين رضا العملاء عن وقت التوصيل والمندوب"
}}

**Example 2: BUBBLE CHART (HEATMAP STYLE)**

{{
  "chart_type": "bubble",
  "data": {{
    "datasets": [{{
      "label": "كثافة الاستجابات",
      "data": [
        {{"x": 5, "y": 5, "r": 35}},
        {{"x": 5, "y": 4, "r": 8}},
        {{"x": 4, "y": 5, "r": 12}},
        {{"x": 4, "y": 4, "r": 6}},
        {{"x": 3, "y": 3, "r": 4}}
      ],
      "backgroundColor": [
        "rgba(34, 197, 94, 0.7)",
        "rgba(59, 130, 246, 0.7)",
        "rgba(59, 130, 246, 0.7)",
        "rgba(251, 191, 36, 0.7)",
        "rgba(251, 191, 36, 0.7)"
      ],
      "borderColor": [
        "rgba(22, 163, 74, 1)",
        "rgba(37, 99, 235, 1)",
        "rgba(37, 99, 235, 1)",
        "rgba(245, 158, 11, 1)",
        "rgba(245, 158, 11, 1)"
      ],
      "borderWidth": 3,
      "hoverBackgroundColor": "rgba(102, 126, 234, 0.9)",
      "hoverBorderColor": "#fff",
      "hoverBorderWidth": 4
    }}]
  }},
  "options": {{
    "responsive": true,
    "maintainAspectRatio": false,
    "scales": {{
      "x": {{
        "title": {{
          "display": true,
          "text": "رضا عن وقت التوصيل",
          "font": {{"size": 14, "weight": "bold"}},
          "color": "#374151"
        }},
        "min": 0,
        "max": 6,
        "ticks": {{
          "stepSize": 1,
          "callback": "function(value) {{ const labels = ['', 'ضعيف جداً', 'ضعيف', 'مقبول', 'جيد', 'ممتاز', '']; return labels[value] || ''; }}",
          "font": {{"size": 12}},
          "color": "#6b7280"
        }},
        "grid": {{
          "color": "rgba(0, 0, 0, 0.08)",
          "lineWidth": 1
        }}
      }},
      "y": {{
        "title": {{
          "display": true,
          "text": "رضا عن عامل التوصيل",
          "font": {{"size": 14, "weight": "bold"}},
          "color": "#374151"
        }},
        "min": 0,
        "max": 6,
        "ticks": {{
          "stepSize": 1,
          "callback": "function(value) {{ const labels = ['', 'ضعيف جداً', 'ضعيف', 'مقبول', 'جيد', 'ممتاز', '']; return labels[value] || ''; }}",
          "font": {{"size": 12}},
          "color": "#6b7280"
        }},
        "grid": {{
          "color": "rgba(0, 0, 0, 0.08)",
          "lineWidth": 1
        }}
      }}
    }},
    "plugins": {{
      "legend": {{
        "display": true,
        "position": "top",
        "labels": {{
          "font": {{"size": 13, "weight": "600"}},
          "color": "#374151",
          "padding": 15
        }}
      }},
      "tooltip": {{
        "enabled": true,
        "backgroundColor": "rgba(0, 0, 0, 0.85)",
        "padding": 12,
        "cornerRadius": 8,
        "callbacks": {{
          "label": "function(context) {{ return 'عدد المستجيبين: ' + (context.raw.r * 10); }}"
        }}
      }}
    }}
  }},
  "title": "الخريطة الحرارية للارتباطات - كثافة الاستجابات"
}}

═══════════════════════════════════════════════════════════════════════════════
✅ PRE-SUBMISSION VALIDATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

Before you submit your response, VERIFY EVERY ITEM:

**FORMAT CHECKS:**
□ Response starts with [ (opening bracket)
□ Response ends with ] (closing bracket)
□ NO markdown code blocks (no ```json)
□ NO explanatory text before or after JSON
□ NO comments in the JSON
□ Valid JSON syntax (run through mental JSON validator)

**CONTENT CHECKS:**
□ Generated MAXIMUM 2 different charts
□ EVERY chart MUST have a "chart_type" field (REQUIRED - cannot be missing!)
□ chart_type must be one of: "bar", "line", "doughnut", "pie", "scatter", "bubble", "radar", "wordcloud"
□ Each chart has unique "chart_type"
□ Each chart tells a different story
□ At least one scatter OR bubble chart included
□ All chart types are professional and complement each other

**DATA ACCURACY:**
□ All data values match the source correlation data
□ ResponseCount values are accurate
□ Percentages are correctly calculated
□ Labels match actual response categories

**ARABIC REQUIREMENTS:**
□ ALL titles are in Arabic
□ ALL labels are in Arabic
□ ALL tooltip text is in Arabic
□ ALL legend text is in Arabic

**QUESTION TEXT:**
□ Actual question text extracted from column names
□ NO placeholders like [Question_1], Q1, Q2
□ Question text shortened appropriately
□ Question text makes sense in context

**VISUAL QUALITY:**
□ Colors follow sentiment rules
□ Hover effects configured
□ Tooltips are informative
□ Fonts are properly sized
□ Borders and spacing look professional

**CHART-SPECIFIC:**
□ Scatter/Bubble: Points sized based on ResponseCount
□ Bar charts: Horizontal for long labels, vertical for short
□ Doughnuts: Cutout configured, colors distinct
□ All axes have proper titles and ticks

═══════════════════════════════════════════════════════════════════════════════
📊 YOUR CORRELATION DATA
═══════════════════════════════════════════════════════════════════════════════

{{CORRELATION_DATA}}

═══════════════════════════════════════════════════════════════════════════════
🎯 YOUR MISSION
═══════════════════════════════════════════════════════════════════════════════

**STEP 1: ANALYZE THE DATA**
- Identify the two questions being correlated
- Extract clean Arabic question text
- Note the response patterns (top combinations, outliers)
- Calculate key insights (correlation strength, agreement rate)

**STEP 2: PLAN YOUR CHARTS (MAXIMUM 2)**
Choose the 2 most insightful charts that best illustrate the relationship:
- Scatter plot showing correlation pattern
- Bubble chart (heatmap style) showing density
- Grouped bar comparing distributions
- Top combinations horizontal bar
- Dual doughnut distributions

**STEP 3: GENERATE CHART CONFIGURATIONS**
- Use actual question text in titles
- Apply appropriate colors (sentiment-based)
- Configure professional styling
- Add informative tooltips
- Ensure data accuracy

**STEP 4: VALIDATE AGAINST CHECKLIST**
Go through each checkbox systematically

**STEP 5: RETURN PURE JSON**
- Start with [
- End with ]
- Nothing else

═══════════════════════════════════════════════════════════════════════════════

NOW GENERATE THE ELITE CORRELATION VISUALIZATION JSON (MAX 2 CHARTS).
"""
    system_content = system_content.replace("{{CORRELATION_DATA}}", correlation_data)

    human_content = """Generate MAXIMUM OF 2 PROFESSIONAL correlation visualization chart configurations NOW.

CRITICAL REQUIREMENTS (NON-NEGOTIABLE):
✅ Generate MAXIMUM 2 DIVERSE charts (scatter, bubble, bar, doughnut, etc.)
✅ Use ACTUAL question text from column names in ALL titles
✅ Return ONLY pure JSON array (no markdown, no text)
✅ Start with [ and end with ]
✅ ALL text in Arabic (titles, labels, tooltips)
✅ NO placeholders ([Question_1], Q1, Q2, etc.)
✅ Professional styling with gradients and hover effects
✅ Each chart tells a DIFFERENT story

REMEMBER:
- First chart should be the most insightful (usually scatter or bubble)
- Second chart must complement the first, NOT duplicate its story (consider bar or doughnut or horizontal bar)
- Use sentiment-based colors
- Make tooltips informative
- Validate against the checklist
- NEVER generate more than 2 chart objects!

Generate the JSON array now (START WITH [, END WITH ]):"""

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]


class CorrelationVisualizationGeneratorPrompt:
    """Wrapper class to maintain .invoke() compatibility"""
    def invoke(self, inputs):
        correlation_data = inputs.get("CORRELATION_DATA", "")
        questions_text = inputs.get("QUESTIONS_TEXT", "")
        
        prompt_messages = _create_correlation_visualization_prompt(correlation_data)
        
        # If questions text is provided, add it to emphasize using actual question text
        if questions_text:
            question_context = f"""

═══════════════════════════════════════════════════════════════════════════════
📋 ACTUAL QUESTION TEXT (USE IN CHART TITLES)
═══════════════════════════════════════════════════════════════════════════════

**THE ACTUAL SURVEY QUESTIONS BEING CORRELATED:**
{questions_text}

**CRITICAL: USE THE ACTUAL QUESTION TEXT IN CHART TITLES**
- Extract question text from column names in the correlation data
- If column names don't contain full question text, use the questions listed above
- NEVER use placeholders like "السؤال الأول", "Question 1", "Q1", "[Question_1]"
- ALWAYS use the actual question text or a meaningful abbreviation of it

═══════════════════════════════════════════════════════════════════════════════
"""
            # Add the question context before "YOUR MISSION"
            if len(prompt_messages) >= 2:
                original_system_content = prompt_messages[0].content
                original_system_content = original_system_content.replace(
                    "═══════════════════════════════════════════════════════════════════════════════\n🎯 YOUR MISSION",
                    question_context + "\n═══════════════════════════════════════════════════════════════════════════════\n🎯 YOUR MISSION"
                )
                prompt_messages[0].content = original_system_content
        
        return prompt_messages

# Create the instance for correlation visualization
visualization_generator_for_correlation_prompt = CorrelationVisualizationGeneratorPrompt()