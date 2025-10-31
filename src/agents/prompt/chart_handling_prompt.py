from langchain_core.messages import SystemMessage, HumanMessage

# Using a function-based approach to avoid template variable conflicts
# This allows us to use normal single brackets { } in the prompt text
def chart_handling_prompt_function(charts):
    """
    Creates a chart handling prompt with the given charts content.
    Returns a list of messages for the LLM.
    """
    # Using regular string to avoid escaping { and }
    system_content = """---

CHART HANDLING INSTRUCTIONS

🚨 CRITICAL OUTPUT REQUIREMENT 🚨
You MUST return ONLY raw JSON without any escaping. NO text before or after the JSON.

❌ FORBIDDEN:
- Any text, explanations, or markdown outside the JSON array
- Code block markers (```json or ```)
- Wrapping in code blocks or strings
- Double escaping quotes or special characters
- Anything that is not valid JSON
- Markdown formatting
- Explanatory text

✅ REQUIRED:
- Start response with [
- End response with ]
- Valid JSON array containing visualization objects
- Properly escaped JSON strings (use \\n for newlines, \\" for quotes within strings)
- Direct JSON output only
- ALL JavaScript objects MUST have commas between properties

---

When charts are provided in the analysis results:

1. **Extract and Include**: Copy the complete chart HTML/JS code into the visualizations array
2. **Preserve Structure**: Maintain all HTML, CSS, and JavaScript exactly as provided
3. **Proper JSON Escaping**: Use proper JSON escaping (\\n for newlines, \\" for quotes)
4. **Single Container**: Combine all charts into ONE container with a unified layout
5. **JavaScript Syntax**: ENSURE ALL JavaScript objects have commas between properties (count,type not count type)
6. **Add Context**: Provide a chart_description that:
   - Explains what the visualization shows
   - Highlights the key business insight
   - Connects to the narrative in detailed_analysis
7. **Reference in Analysis**: Mention the chart naturally in your detailed_analysis where relevant

**Chart JSON Rules:**
- USE proper JSON escaping: \\n for newlines, \\" for quotes within strings
- Keep all HTML, CSS, and JavaScript logic intact
- Preserve all formatting within the HTML string
- Return valid JSON structure
- Test the final JSON structure for validity
- CRITICAL: Check all JavaScript objects for missing commas between properties

**JavaScript Syntax Requirements:**
- ALL object properties MUST be separated by commas
- CORRECT: {text:'word', count:50, type:'positive'}
- INCORRECT: {text:'word', count:50 type:'positive'}  ← Missing comma!
- CORRECT: {label:'Data', data:[10,20], backgroundColor:'#4e73df'}
- INCORRECT: {label:'Data' data:[10,20] backgroundColor:'#4e73df'}  ← Missing commas!

**Single Container Structure:**
All charts must be wrapped in ONE parent container with:
- Shared Chart.js and D3.js script imports (load once at the top)
- Unified styling for consistent appearance
- Grid or flex layout for multiple charts
- Unique canvas/div IDs for each chart (chart1, chart2, etc.)
- Responsive design that adapts to screen size

**CORRECT OUTPUT FORMAT - Your response must look EXACTLY like this:**

[
  {
    "chart_html": "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>\\n<script src='https://d3js.org/d3.v6.min.js'></script>\\n<style>\\n.charts-container {\\n  display: grid;\\n  grid-template-columns: 1fr 1fr;\\n  gap: 20px;\\n  padding: 20px;\\n}\\n.chart-widget {\\n  background: #fff;\\n  border-radius: 8px;\\n  padding: 20px;\\n  box-shadow: 0 2px 4px rgba(0,0,0,0.1);\\n}\\n</style>\\n<div class='charts-container'>\\n  <div class='chart-widget'>\\n    <canvas id='chart1'></canvas>\\n  </div>\\n  <div class='chart-widget'>\\n    <canvas id='chart2'></canvas>\\n  </div>\\n</div>\\n<script>\\nnew Chart(document.getElementById('chart1'), {\\n  type: 'bar',\\n  data: {\\n    labels: ['A', 'B', 'C'],\\n    datasets: [{\\n      label: 'Data',\\n      data: [10, 20, 30],\\n      backgroundColor: '#4e73df'\\n    }]\\n  },\\n  options: {\\n    responsive: true\\n  }\\n});\\nnew Chart(document.getElementById('chart2'), {\\n  type: 'pie',\\n  data: {\\n    labels: ['X', 'Y'],\\n    datasets: [{\\n      data: [60, 40],\\n      backgroundColor: ['#1cc88a', '#36b9cc']\\n    }]\\n  }\\n});\\n</script>",
    "chart_description": "Description of the combined visualization showing both bar and pie charts"
  }
]

Note: The response is a JSON ARRAY containing ONE object, not just the object alone.
Note: NO markdown code blocks, NO explanations, NO wrapping - just raw JSON starting with [ and ending with ].
Note: ALL JavaScript object properties MUST have commas between them!

---

VISUALIZATION FIELD REQUIREMENTS

**visualizations**: Array with ONE object containing all charts in a single container; use empty array [] if no charts provided

The single visualization object must contain:
- **chart_html**: Complete HTML string with ALL charts in one container, shared libraries, unified styling
- **chart_description**: Comprehensive description covering all charts and their collective insights

---

CHART INTEGRATION IN ANALYSIS

- Reference charts naturally within your detailed_analysis narrative
- Connect visualization insights to the business story
- Use phrases like "The accompanying dashboard illustrates..." or "As shown in the visualizations..."
- Ensure chart descriptions align with the narrative context

---

JAVASCRIPT VALIDATION CHECKLIST

Before finalizing, check EVERY JavaScript object:

**Common Missing Comma Errors**:
□ Check: {text:'word', count:50, type:'positive'}  ← Comma after count!
□ Check: {label:'Data', data:[10,20], backgroundColor:'#fff'}  ← Commas between ALL properties!
□ Check: datasets:[{label:'X', data:[1,2,3], backgroundColor:'#f00'}]  ← All commas present!
□ Verify array elements are comma-separated: [item1, item2, item3]
□ Verify object properties are comma-separated: {prop1: val1, prop2: val2, prop3: val3}

**Example of CORRECT JavaScript syntax in word cloud**:
const words = [
  {text:'excellent', count:150, type:'positive'},
  {text:'good', count:120, type:'positive'},
  {text:'fast', count:100, type:'positive'}
];

**Example of INCORRECT JavaScript syntax (DO NOT USE)**:
const words = [
  {text:'excellent' count:150 type:'positive'},  ← Missing commas!
  {text:'good', count:120 type:'positive'}  ← Missing comma after count!
];

---

CHART QUALITY CHECKLIST

Before finalizing, verify:

**Chart Format**:
□ Response starts with [ and ends with ]
□ ALL charts combined in ONE parent container
□ Chart.js and D3.js libraries loaded once at the top
□ Unique canvas/div IDs for each chart (chart1, chart2, chart3, etc.)
□ Properly escaped JSON (\\n for newlines, \\" for quotes)
□ No double escaping
□ No wrapping in code blocks or strings
□ No markdown formatting
□ No explanatory text outside JSON
□ Grid or flex layout for responsive design
□ Valid JSON structure maintained
□ ALL JavaScript objects have commas between properties

**Chart Integration**:
□ References charts naturally in the narrative
□ Chart description covers all visualizations
□ Visualizations support the main conclusions
□ Empty array [] used when no charts provided
□ Single container ensures consistent presentation

---

Charts Input:
"""
    # Append the charts content manually
    system_content += str(charts)

    human_content = "Respond in Arabic"

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]


# For compatibility with existing code that uses .invoke()
class ChartHandlingPrompt:
    """Wrapper class to maintain .invoke() compatibility"""
    def invoke(self, inputs):
        charts = inputs.get("charts", "")
        return _create_chart_handling_prompt(charts)

# Rename the function to avoid conflict
_create_chart_handling_prompt = chart_handling_prompt_function

# Create the instance with the same name for compatibility
chart_handling_prompt = ChartHandlingPrompt()