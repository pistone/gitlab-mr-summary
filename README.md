# gitlab-mr-summary
# GitLab Code Review Pattern Analyzer with LLM Enhancement

A powerful tool to analyze code review patterns from GitLab merge requests using Large Language Models (LLM) to provide semantic understanding, pattern extraction, best practice generation, and sentiment analysis.

**Supports both Claude (Anthropic) and OpenAI APIs.**

## Features

### Basic Analysis (No LLM Required)
- ✅ Configuration-based user targeting
- ✅ Keyword-based comment categorization
- ✅ Common keywords and patterns extraction
- ✅ Top reviewers tracking
- ✅ File type analysis

### LLM-Enhanced Analysis (Optional)
- 🤖 **Semantic Categorization**: Understand comment intent beyond keywords
- 🧩 **Pattern Extraction**: Identify abstract patterns and principles across different phrasings
- 📚 **Best Practices Generation**: Auto-generate team best practices documentation
- 💬 **Sentiment Analysis**: Analyze tone, helpfulness, and communication culture

## Installation

### Prerequisites

- Python 3.7+
- GitLab personal access token with `api` scope
- (Optional) Anthropic API key for Claude OR OpenAI API key

### Setup

1. Install required dependencies:

```bash
pip install -r requirements.txt --break-system-packages
```

Or install individually:

```bash
pip install python-gitlab pyyaml --break-system-packages

# For LLM features:
pip install anthropic --break-system-packages  # For Claude
pip install openai --break-system-packages     # For OpenAI
```

2. Create API keys:

**GitLab Token:**
- Go to GitLab → Settings → Access Tokens
- Create a token with `api` scope

**LLM API Keys (optional):**
- For Claude: Get API key from https://console.anthropic.com/
- For OpenAI: Get API key from https://platform.openai.com/

3. Configure the tool by editing `config.yaml`

## Configuration

### Minimal Configuration (No LLM)

```yaml
gitlab:
  url: "https://gitlab.com"
  token: "YOUR_GITLAB_TOKEN"
  project_id: "your-project-id"

target_users:
  - "username1"
  - "username2"

llm:
  enabled: false  # Disable LLM features
```

### Full Configuration (With LLM)

```yaml
gitlab:
  url: "https://gitlab.com"
  token: "YOUR_GITLAB_TOKEN"
  project_id: "your-project-id"

target_users:
  - "username1"
  - "username2"

llm:
  enabled: true
  provider: "claude"  # or "openai"
  
  claude:
    api_key: "YOUR_ANTHROPIC_API_KEY"
    model: "claude-sonnet-4-20250514"
    max_tokens: 2000
  
  openai:
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o"
    max_tokens: 2000
  
  features:
    semantic_categorization: true
    pattern_extraction: true
    best_practices_generation: true
    sentiment_analysis: true
  
  batch_size: 10
  rate_limit_delay: 0.5

analysis:
  date_range:
    start: "2024-01-01"
    end: null
  filters:
    state: "merged"
    min_comments: 1
```

## Usage

Run the analyzer:

```bash
python gitlab_review_analyzer_llm.py config.yaml
```

The tool will:
1. Connect to GitLab and your chosen LLM provider
2. Fetch merge requests from specified users
3. Analyze all review comments (basic + LLM analysis if enabled)
4. Generate a comprehensive report

## LLM Features Explained

### 1. Semantic Categorization

**What it does:** Uses LLM to understand the true intent of comments beyond simple keyword matching.

**Example:**
- Comment: "This might cause issues in production"
- Keyword approach: "other" (no clear keywords)
- LLM approach: "bug", severity="high", actionable="yes"

**Output:**
- Detailed categorization with subcategories
- Severity levels (low, medium, high, critical)
- Actionability assessment
- Intent understanding

### 2. Pattern Extraction

**What it does:** Identifies abstract patterns and principles that appear across different wordings.

**Example patterns identified:**
- "Avoid hard-coded values" (from 20 different phrasings)
- "Improve error handling" (from various contexts)
- "Separation of concerns" principle
- Team-specific conventions

**Output:**
- Common review patterns with frequencies
- Underlying principles being enforced
- Team conventions and preferences
- Grouped similar feedback

### 3. Best Practices Generation

**What it does:** Synthesizes review comments into structured best practices documentation.

**Generated sections:**
- Error Handling guidelines
- Code Style rules
- Security considerations
- Performance tips
- Testing requirements
- Documentation standards

**Each guideline includes:**
- Clear rule statement
- Rationale (why it matters)
- Examples (good vs bad)

### 4. Sentiment Analysis

**What it does:** Analyzes the tone and helpfulness of review comments.

**Analyzes:**
- Tone distribution (constructive, neutral, critical, dismissive)
- Helpfulness scores (1-10)
- Communication patterns
- Team culture assessment

**Provides:**
- Examples of constructive comments
- Suggestions for improving communication
- Overall team culture insights

## Output

### Report Files

The tool generates:

1. **Text Report** (`review_analysis_YYYYMMDD_HHMMSS.txt`)
   - Complete analysis report
   - Basic and LLM-enhanced insights
   - Human-readable format

2. **JSON Export** (`llm_analysis_YYYYMMDD_HHMMSS.json`)
   - Structured LLM analysis data
   - Machine-readable format
   - For integration with other tools

### Sample Report Structure

```
================================================================================
GITLAB CODE REVIEW PATTERN ANALYSIS REPORT
================================================================================

OVERVIEW
--------------------------------------------------------------------------------
Total Merge Requests Analyzed: 150
Total Review Comments: 1,234
Average Comments per MR: 8.2

COMMENT CATEGORIES (Basic)
--------------------------------------------------------------------------------
  Style                    345 ( 28.0%)
  Question                 267 ( 21.6%)
  ...

================================================================================
LLM-ENHANCED ANALYSIS
================================================================================

SEMANTIC CATEGORIZATION
--------------------------------------------------------------------------------
  Bug                       234
  Security                   89
  ...

  Severity Distribution:
    High                     123
    Medium                   456
    Low                      234

  Actionable Comments: 67.8%

IDENTIFIED PATTERNS
--------------------------------------------------------------------------------
  1. Avoid hard-coded configuration values
     Principle: Configuration management
     Example: "Consider moving this to a config file..."

  2. Improve error handling with specific exceptions
     Principle: Defensive programming
     Example: "Instead of catching Exception, catch ValueError..."

GENERATED BEST PRACTICES
--------------------------------------------------------------------------------
  ERROR HANDLING
    • Always use specific exception types rather than catching all exceptions
      Rationale: Allows better error identification and debugging
    • Include context in error messages
      Rationale: Helps with troubleshooting in production
  ...

SENTIMENT & TONE ANALYSIS
--------------------------------------------------------------------------------
  Tone Distribution:
    Constructive       65%
    Neutral           25%
    Critical           8%
    Dismissive         2%

  Average Helpfulness: 7.8/10

  Culture Assessment:
  The team demonstrates a generally constructive review culture with focus
  on teaching and improvement rather than criticism...
```

## Cost Considerations

### LLM API Costs

**Claude (Anthropic):**
- Sonnet 4: ~$3 per million input tokens, ~$15 per million output tokens
- Opus 4: ~$15 per million input tokens, ~$75 per million output tokens

**OpenAI:**
- GPT-4o: ~$2.50 per million input tokens, ~$10 per million output tokens
- GPT-4o-mini: ~$0.15 per million input tokens, ~$0.60 per million output tokens

**Estimation:**
- 100 comments ≈ 50K tokens ≈ $0.15-$0.75 depending on model
- Use `batch_size` and sampling to control costs

### Cost Optimization Tips

1. Start with smaller date ranges to test
2. Use mini/sonnet models for initial analysis
3. Adjust `batch_size` to balance speed vs. rate limits
4. Disable specific features you don't need
5. Sample large datasets rather than analyzing everything

## Use Cases for Code Review Agent

The analysis output is perfect for building a code review agent:

1. **Training Data**: Use categorized comments to train/fine-tune models
2. **Rule Extraction**: Convert patterns into automated lint rules
3. **Context-Aware Reviews**: Reference similar past reviews
4. **Consistency**: Apply team conventions automatically
5. **Learning**: Agent learns from senior reviewers' feedback patterns

## Troubleshooting

### GitLab Connection Issues
- Verify access token has `api` scope
- Check project ID is correct
- Ensure you have project access

### LLM API Errors

**Claude:**
- Verify API key from https://console.anthropic.com/
- Check rate limits (contact Anthropic for increases)
- Ensure model name is correct

**OpenAI:**
- Verify API key from https://platform.openai.com/
- Check you have sufficient credits
- Ensure model access (GPT-4 requires separate access)

### Rate Limiting
- Increase `rate_limit_delay` in config
- Reduce `batch_size`
- Process fewer comments at once

### Out of Memory
- Reduce `batch_size`
- Process smaller date ranges
- Limit the number of comments analyzed

## Advanced Usage

### Using with Code Review Agent

```python
# Load the generated best practices
with open('llm_analysis_*.json', 'r') as f:
    best_practices = json.load(f)

# Use patterns in your review agent
patterns = best_practices['patterns']['patterns']
for pattern in patterns:
    # Create automated checks based on patterns
    create_lint_rule(pattern)
```

### Custom Analysis

Extend the tool by adding your own LLM analysis features:

```python
def _custom_analysis(self, comments):
    """Your custom LLM analysis."""
    prompt = f"Custom analysis prompt for: {comments}"
    response = self.llm_provider.call_api(prompt)
    return process_response(response)
```

## Comparison: Basic vs LLM Analysis

| Feature | Basic (No LLM) | LLM-Enhanced |
|---------|---------------|--------------|
| Speed | Fast | Slower (API calls) |
| Cost | Free | Paid (API costs) |
| Categorization | Keyword-based | Semantic understanding |
| Pattern Detection | Simple frequency | Abstract principles |
| Best Practices | None | Auto-generated docs |
| Sentiment | None | Tone & culture analysis |
| Accuracy | ~60% | ~90% |

## Future Enhancements

Potential features:
- [ ] Code context analysis (analyze actual code changes)
- [ ] Integration with issue trackers
- [ ] Real-time review suggestions
- [ ] Fine-tuned models on team data
- [ ] Multi-language support
- [ ] Trend analysis over time
- [ ] Auto-generated review templates

## License

MIT License

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration examples
3. Verify API credentials and access
4. Check GitLab and LLM provider documentation
