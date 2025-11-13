#!/usr/bin/env python3
"""
GitLab Code Review Pattern Analyzer with LLM Enhancement

This tool analyzes code review patterns from GitLab merge requests using LLM
to provide semantic understanding, pattern extraction, best practice generation,
and sentiment analysis.

Supports both Claude (Anthropic) and OpenAI APIs.
"""

import yaml
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
    
    def call_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call the LLM API with a prompt."""
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    """Claude API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config['claude']['api_key']
            )
            self.model = config['claude']['model']
            self.max_tokens = config['claude'].get('max_tokens', 2000)
        except ImportError:
            print("Error: anthropic library not installed")
            print("Install it with: pip install anthropic --break-system-packages")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Missing Claude configuration: {e}")
            sys.exit(1)
    
    def call_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call Claude API."""
        try:
            time.sleep(self.rate_limit_delay)
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system=system_prompt if system_prompt else ""
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config['openai']['api_key']
            )
            self.model = config['openai']['model']
            self.max_tokens = config['openai'].get('max_tokens', 2000)
        except ImportError:
            print("Error: openai library not installed")
            print("Install it with: pip install openai --break-system-packages")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Missing OpenAI configuration: {e}")
            sys.exit(1)
    
    def call_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call OpenAI API."""
        try:
            time.sleep(self.rate_limit_delay)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise


class GitLabReviewAnalyzer:
    """Analyzes GitLab merge request reviews with LLM enhancement."""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration file."""
        self.config = self._load_config(config_path)
        self.gl = None
        self.project = None
        self.llm_provider = None
        
        # Initialize LLM provider if enabled
        if self.config.get('llm', {}).get('enabled', False):
            self._initialize_llm()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            if 'gitlab' not in config:
                raise ValueError("Missing 'gitlab' section in config")
            if 'target_users' not in config or not config['target_users']:
                raise ValueError("Missing or empty 'target_users' list in config")
            
            print(f"✓ Configuration loaded successfully")
            print(f"  Target users: {', '.join(config['target_users'])}")
            
            # Check LLM config
            if config.get('llm', {}).get('enabled', False):
                provider = config['llm']['provider']
                print(f"  LLM enhancement: enabled (provider: {provider})")
            else:
                print(f"  LLM enhancement: disabled")
            
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML in configuration file: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def _initialize_llm(self):
        """Initialize the LLM provider."""
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'claude')
        
        if provider == 'claude':
            self.llm_provider = ClaudeProvider(llm_config)
            print("✓ Claude API initialized")
        elif provider == 'openai':
            self.llm_provider = OpenAIProvider(llm_config)
            print("✓ OpenAI API initialized")
        else:
            print(f"Error: Unknown LLM provider: {provider}")
            sys.exit(1)
    
    def connect_gitlab(self):
        """Connect to GitLab instance."""
        try:
            import gitlab
        except ImportError:
            print("Error: python-gitlab library not installed")
            print("Install it with: pip install python-gitlab --break-system-packages")
            sys.exit(1)
        
        try:
            gitlab_config = self.config['gitlab']
            self.gl = gitlab.Gitlab(
                gitlab_config['url'],
                private_token=gitlab_config['token']
            )
            self.gl.auth()
            
            # Get project
            self.project = self.gl.projects.get(gitlab_config['project_id'])
            print(f"✓ Connected to GitLab project: {self.project.name}")
            
        except gitlab.exceptions.GitlabAuthenticationError:
            print("Error: GitLab authentication failed. Check your token.")
            sys.exit(1)
        except gitlab.exceptions.GitlabGetError as e:
            print(f"Error: Could not access project. {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error connecting to GitLab: {e}")
            sys.exit(1)
    
    def fetch_merge_requests(self) -> List[Any]:
        """Fetch merge requests for target users."""
        print("\nFetching merge requests...")
        
        target_users = self.config['target_users']
        filters = self.config.get('analysis', {}).get('filters', {})
        date_range = self.config.get('analysis', {}).get('date_range', {})
        
        all_mrs = []
        
        # Prepare filters
        mr_filters = {
            'state': filters.get('state', 'merged'),
            'scope': 'all',
            'order_by': 'updated_at',
            'sort': 'desc'
        }
        
        # Add date filters if specified
        if date_range.get('start'):
            mr_filters['updated_after'] = date_range['start']
        if date_range.get('end'):
            mr_filters['updated_before'] = date_range['end']
        
        # Fetch MRs
        for username in target_users:
            print(f"  Fetching MRs for user: {username}")
            mr_filters['author_username'] = username
            
            try:
                mrs = self.project.mergerequests.list(
                    get_all=True,
                    **mr_filters
                )
                
                # Filter by minimum comments if specified
                min_comments = filters.get('min_comments', 1)
                mrs = [mr for mr in mrs if mr.user_notes_count >= min_comments]
                
                all_mrs.extend(mrs)
                print(f"    Found {len(mrs)} MRs")
                
            except Exception as e:
                print(f"    Error fetching MRs for {username}: {e}")
        
        print(f"\n✓ Total MRs fetched: {len(all_mrs)}")
        return all_mrs
    
    def analyze_comments(self, merge_requests: List[Any]) -> Dict[str, Any]:
        """Analyze comments from merge requests."""
        print("\nAnalyzing review comments...")
        
        analysis = {
            'total_mrs': len(merge_requests),
            'total_comments': 0,
            'comment_categories': defaultdict(int),
            'common_keywords': Counter(),
            'file_types': Counter(),
            'review_patterns': [],
            'top_reviewers': Counter(),
            'avg_comments_per_mr': 0,
            'llm_analysis': {}
        }
        
        all_comments = []
        
        for i, mr in enumerate(merge_requests, 1):
            if i % 10 == 0:
                print(f"  Processing MR {i}/{len(merge_requests)}...")
            
            try:
                # Get all notes (comments) for this MR
                notes = mr.notes.list(get_all=True)
                
                for note in notes:
                    # Skip system notes
                    if note.system:
                        continue
                    
                    comment_data = {
                        'text': note.body,
                        'author': note.author['username'],
                        'mr_iid': mr.iid,
                        'mr_title': mr.title,
                        'created_at': note.created_at
                    }
                    
                    all_comments.append(comment_data)
                    analysis['total_comments'] += 1
                    analysis['top_reviewers'][note.author['username']] += 1
                    
                    # Basic categorization (will be enhanced by LLM if enabled)
                    category = self._categorize_comment(note.body.lower())
                    analysis['comment_categories'][category] += 1
                    
                    # Extract keywords
                    keywords = self._extract_keywords(note.body.lower())
                    analysis['common_keywords'].update(keywords)
                
                # Track file types from changes
                try:
                    changes = mr.changes()
                    for change in changes.get('changes', []):
                        file_path = change.get('new_path', '')
                        if file_path:
                            ext = file_path.split('.')[-1] if '.' in file_path else 'no_ext'
                            analysis['file_types'][ext] += 1
                except:
                    pass
                    
            except Exception as e:
                print(f"    Error processing MR {mr.iid}: {e}")
        
        # Calculate averages
        if len(merge_requests) > 0:
            analysis['avg_comments_per_mr'] = analysis['total_comments'] / len(merge_requests)
        
        # LLM-enhanced analysis if enabled
        if self.llm_provider and all_comments:
            print("\n" + "="*80)
            print("Running LLM-Enhanced Analysis...")
            print("="*80)
            analysis['llm_analysis'] = self._llm_enhanced_analysis(all_comments)
        
        print(f"✓ Analysis complete")
        return analysis
    
    def _categorize_comment(self, comment: str) -> str:
        """Basic categorization (fallback if LLM disabled)."""
        comment_lower = comment.lower()
        
        categories = {
            'bug': ['bug', 'error', 'issue', 'problem', 'broken', 'fix', 'crash'],
            'security': ['security', 'vulnerability', 'sql injection', 'xss', 'auth', 'permission'],
            'performance': ['performance', 'slow', 'optimize', 'efficient', 'memory', 'speed'],
            'style': ['style', 'formatting', 'indent', 'naming', 'convention', 'lint'],
            'architecture': ['architecture', 'design', 'pattern', 'structure', 'refactor', 'abstraction'],
            'testing': ['test', 'coverage', 'unit test', 'integration', 'mock'],
            'documentation': ['doc', 'comment', 'readme', 'documentation', 'explain'],
            'question': ['?', 'why', 'what', 'how', 'clarify', 'wondering'],
            'approval': ['lgtm', 'looks good', 'approved', 'great', 'nice', '👍'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in comment_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _extract_keywords(self, comment: str) -> List[str]:
        """Extract meaningful keywords from a comment."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'we', 'they', 'it', 'what', 'which', 'who', 'when', 'where'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', comment.lower())
        return [w for w in words if w not in stop_words]
    
    def _llm_enhanced_analysis(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform LLM-enhanced analysis on comments."""
        features = self.config.get('llm', {}).get('features', {})
        llm_results = {}
        
        # Feature 1: Semantic Categorization
        if features.get('semantic_categorization', False):
            print("\n1. Semantic Categorization...")
            llm_results['semantic_categories'] = self._semantic_categorization(comments)
        
        # Feature 2: Pattern Extraction
        if features.get('pattern_extraction', False):
            print("\n2. Pattern Extraction...")
            llm_results['patterns'] = self._pattern_extraction(comments)
        
        # Feature 3: Best Practices Generation
        if features.get('best_practices_generation', False):
            print("\n3. Best Practices Generation...")
            llm_results['best_practices'] = self._best_practices_generation(comments)
        
        # Feature 4: Sentiment Analysis
        if features.get('sentiment_analysis', False):
            print("\n4. Sentiment Analysis...")
            llm_results['sentiment'] = self._sentiment_analysis(comments)
        
        return llm_results
    
    def _semantic_categorization(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to semantically categorize comments."""
        print("  Analyzing comment semantics...")
        
        batch_size = self.config.get('llm', {}).get('batch_size', 10)
        categorized = []
        
        # Process in batches
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i+batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
            
            # Create prompt
            comments_text = "\n\n".join([
                f"Comment {j+1}:\n{comment['text']}"
                for j, comment in enumerate(batch)
            ])
            
            prompt = f"""Analyze these code review comments and categorize each one.

{comments_text}

For each comment, provide:
1. Primary category (bug, security, performance, style, architecture, testing, documentation, question, suggestion, approval)
2. Subcategory (more specific classification)
3. Severity (low, medium, high, critical)
4. Intent (what is the reviewer trying to achieve?)
5. Actionable (yes/no - is this a concrete action item?)

Return as JSON array with one object per comment:
[{{"comment_id": 1, "category": "...", "subcategory": "...", "severity": "...", "intent": "...", "actionable": "yes/no"}}]

Return ONLY the JSON array, no other text."""
            
            try:
                response = self.llm_provider.call_api(prompt)
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    batch_results = json.loads(json_match.group())
                    # Add original comment data
                    for j, result in enumerate(batch_results):
                        if j < len(batch):
                            result['original_comment'] = batch[j]['text']
                            result['author'] = batch[j]['author']
                    categorized.extend(batch_results)
            except Exception as e:
                print(f"    Error processing batch: {e}")
        
        # Aggregate results
        category_counts = Counter()
        severity_counts = Counter()
        actionable_count = 0
        
        for item in categorized:
            category_counts[item.get('category', 'unknown')] += 1
            severity_counts[item.get('severity', 'unknown')] += 1
            if item.get('actionable', '').lower() == 'yes':
                actionable_count += 1
        
        return {
            'categorized_comments': categorized,
            'category_distribution': dict(category_counts),
            'severity_distribution': dict(severity_counts),
            'actionable_percentage': (actionable_count / len(categorized) * 100) if categorized else 0
        }
    
    def _pattern_extraction(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract abstract patterns from comments."""
        print("  Extracting review patterns...")
        
        # Sample comments for pattern analysis (avoid sending too many)
        sample_size = min(100, len(comments))
        sample_comments = comments[:sample_size]
        
        comments_text = "\n\n".join([
            f"- {comment['text']}"
            for comment in sample_comments
        ])
        
        prompt = f"""Analyze these code review comments and identify recurring abstract patterns and principles.

{comments_text}

Identify:
1. Common review patterns (e.g., "avoid hard-coded values", "improve error handling")
2. Underlying principles being enforced (e.g., "separation of concerns", "defensive programming")
3. Team-specific conventions or preferences
4. Categories of similar feedback with different wording

Group similar feedback together even if worded differently.
Return as JSON:
{{
  "patterns": [
    {{
      "pattern": "pattern name",
      "principle": "underlying principle",
      "frequency": "estimated frequency",
      "examples": ["example 1", "example 2"]
    }}
  ],
  "conventions": ["convention 1", "convention 2"],
  "summary": "brief summary of key patterns"
}}

Return ONLY valid JSON, no other text."""
        
        try:
            response = self.llm_provider.call_api(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"    Error extracting patterns: {e}")
        
        return {"patterns": [], "conventions": [], "summary": ""}
    
    def _best_practices_generation(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate best practices documentation from comments."""
        print("  Generating best practices...")
        
        # Sample comments
        sample_size = min(100, len(comments))
        sample_comments = comments[:sample_size]
        
        comments_text = "\n\n".join([
            f"- {comment['text']}"
            for comment in sample_comments
        ])
        
        prompt = f"""Based on these code review comments, generate a best practices guide for this team.

{comments_text}

Create a structured guide with:
1. Error Handling best practices
2. Code Style guidelines
3. Security considerations
4. Performance tips
5. Testing requirements
6. Documentation standards

For each section, provide:
- Clear guideline statement
- Rationale (why this matters)
- Example (good vs bad if applicable)

Return as JSON:
{{
  "sections": [
    {{
      "title": "section name",
      "guidelines": [
        {{
          "rule": "guideline statement",
          "rationale": "why this matters",
          "examples": {{"good": "...", "bad": "..."}}
        }}
      ]
    }}
  ],
  "summary": "executive summary"
}}

Return ONLY valid JSON, no other text."""
        
        try:
            response = self.llm_provider.call_api(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"    Error generating best practices: {e}")
        
        return {"sections": [], "summary": ""}
    
    def _sentiment_analysis(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment and tone of comments."""
        print("  Analyzing sentiment and tone...")
        
        # Sample comments
        sample_size = min(100, len(comments))
        sample_comments = comments[:sample_size]
        
        comments_text = "\n\n".join([
            f"Comment {i+1}: {comment['text']}"
            for i, comment in enumerate(sample_comments)
        ])
        
        prompt = f"""Analyze the tone and helpfulness of these code review comments.

{comments_text}

For the overall set of comments, determine:
1. Tone distribution (constructive, neutral, critical, dismissive)
2. Helpfulness score (1-10)
3. Communication patterns (direct, suggestive, questioning)
4. Areas for improvement in review communication

Also identify:
- Most constructive comments (examples)
- Comments that could be improved (examples with suggestions)
- Overall team communication culture

Return as JSON:
{{
  "tone_distribution": {{"constructive": %, "neutral": %, "critical": %, "dismissive": %}},
  "avg_helpfulness": 7.5,
  "communication_style": "description",
  "constructive_examples": ["example 1", "example 2"],
  "improvement_opportunities": [
    {{"comment": "...", "suggestion": "better phrasing: ..."}}
  ],
  "culture_assessment": "assessment text"
}}

Return ONLY valid JSON, no other text."""
        
        try:
            response = self.llm_provider.call_api(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"    Error analyzing sentiment: {e}")
        
        return {}
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive report from the analysis."""
        report_lines = [
            "=" * 80,
            "GITLAB CODE REVIEW PATTERN ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Users: {', '.join(self.config['target_users'])}",
            f"LLM Provider: {self.config.get('llm', {}).get('provider', 'None')}",
            "",
            "OVERVIEW",
            "-" * 80,
            f"Total Merge Requests Analyzed: {analysis['total_mrs']}",
            f"Total Review Comments: {analysis['total_comments']}",
            f"Average Comments per MR: {analysis['avg_comments_per_mr']:.1f}",
            "",
        ]
        
        # Basic categories
        report_lines.extend([
            "COMMENT CATEGORIES (Basic)",
            "-" * 80,
        ])
        
        sorted_categories = sorted(
            analysis['comment_categories'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for category, count in sorted_categories:
            percentage = (count / analysis['total_comments'] * 100) if analysis['total_comments'] > 0 else 0
            report_lines.append(f"  {category.capitalize():20} {count:5} ({percentage:5.1f}%)")
        
        # LLM Analysis Section
        if 'llm_analysis' in analysis and analysis['llm_analysis']:
            report_lines.extend([
                "",
                "=" * 80,
                "LLM-ENHANCED ANALYSIS",
                "=" * 80,
            ])
            
            llm = analysis['llm_analysis']
            
            # Semantic Categorization
            if 'semantic_categories' in llm:
                report_lines.extend([
                    "",
                    "SEMANTIC CATEGORIZATION",
                    "-" * 80,
                ])
                
                sem_cat = llm['semantic_categories']
                if 'category_distribution' in sem_cat:
                    for cat, count in sorted(sem_cat['category_distribution'].items(), key=lambda x: x[1], reverse=True):
                        report_lines.append(f"  {cat.capitalize():20} {count:5}")
                
                if 'severity_distribution' in sem_cat:
                    report_lines.append("\n  Severity Distribution:")
                    for sev, count in sorted(sem_cat['severity_distribution'].items(), key=lambda x: x[1], reverse=True):
                        report_lines.append(f"    {sev.capitalize():15} {count:5}")
                
                if 'actionable_percentage' in sem_cat:
                    report_lines.append(f"\n  Actionable Comments: {sem_cat['actionable_percentage']:.1f}%")
            
            # Patterns
            if 'patterns' in llm:
                report_lines.extend([
                    "",
                    "IDENTIFIED PATTERNS",
                    "-" * 80,
                ])
                
                patterns = llm['patterns']
                if 'patterns' in patterns and patterns['patterns']:
                    for i, pattern in enumerate(patterns['patterns'][:10], 1):
                        report_lines.append(f"\n  {i}. {pattern.get('pattern', 'Unknown')}")
                        report_lines.append(f"     Principle: {pattern.get('principle', 'N/A')}")
                        if 'examples' in pattern and pattern['examples']:
                            report_lines.append(f"     Example: {pattern['examples'][0][:80]}...")
                
                if 'summary' in patterns:
                    report_lines.append(f"\n  Summary: {patterns['summary']}")
            
            # Best Practices
            if 'best_practices' in llm:
                report_lines.extend([
                    "",
                    "GENERATED BEST PRACTICES",
                    "-" * 80,
                ])
                
                bp = llm['best_practices']
                if 'sections' in bp:
                    for section in bp['sections']:
                        report_lines.append(f"\n  {section.get('title', 'Unknown').upper()}")
                        if 'guidelines' in section:
                            for guideline in section['guidelines'][:3]:  # Show top 3
                                report_lines.append(f"    • {guideline.get('rule', 'N/A')}")
                                if 'rationale' in guideline:
                                    report_lines.append(f"      Rationale: {guideline['rationale']}")
                
                if 'summary' in bp:
                    report_lines.append(f"\n  Executive Summary:")
                    report_lines.append(f"  {bp['summary']}")
            
            # Sentiment
            if 'sentiment' in llm:
                report_lines.extend([
                    "",
                    "SENTIMENT & TONE ANALYSIS",
                    "-" * 80,
                ])
                
                sent = llm['sentiment']
                if 'tone_distribution' in sent:
                    report_lines.append("  Tone Distribution:")
                    for tone, pct in sent['tone_distribution'].items():
                        report_lines.append(f"    {tone.capitalize():15} {pct}%")
                
                if 'avg_helpfulness' in sent:
                    report_lines.append(f"\n  Average Helpfulness: {sent['avg_helpfulness']}/10")
                
                if 'culture_assessment' in sent:
                    report_lines.append(f"\n  Culture Assessment:")
                    report_lines.append(f"  {sent['culture_assessment']}")
        
        # Top Reviewers
        report_lines.extend([
            "",
            "TOP REVIEWERS",
            "-" * 80,
        ])
        
        for reviewer, count in analysis['top_reviewers'].most_common(10):
            report_lines.append(f"  {reviewer:30} {count:5} comments")
        
        # Common Keywords
        report_lines.extend([
            "",
            "MOST COMMON KEYWORDS",
            "-" * 80,
        ])
        
        for keyword, count in analysis['common_keywords'].most_common(20):
            report_lines.append(f"  {keyword:30} {count:5}")
        
        # File Types
        report_lines.extend([
            "",
            "FILE TYPES REVIEWED",
            "-" * 80,
        ])
        
        for file_type, count in analysis['file_types'].most_common(15):
            report_lines.append(f"  .{file_type:20} {count:5} changes")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])
        
        return "\n".join(report_lines)
    
    def run(self):
        """Run the complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("GitLab Code Review Pattern Analyzer with LLM Enhancement")
        print("=" * 80)
        
        # Connect to GitLab
        self.connect_gitlab()
        
        # Fetch merge requests
        merge_requests = self.fetch_merge_requests()
        
        if not merge_requests:
            print("\nNo merge requests found for the specified users.")
            return
        
        # Analyze comments
        analysis = self.analyze_comments(merge_requests)
        
        # Generate and display report
        report = self.generate_report(analysis)
        print("\n" + report)
        
        # Save report to file
        report_filename = f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_path = f"/mnt/user-data/outputs/{report_filename}"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {report_filename}")
        
        # Save LLM analysis as JSON if available
        if 'llm_analysis' in analysis and analysis['llm_analysis']:
            json_filename = f"llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = f"/mnt/user-data/outputs/{json_filename}"
            
            with open(json_path, 'w') as f:
                json.dump(analysis['llm_analysis'], f, indent=2)
            
            print(f"✓ LLM analysis saved to: {json_filename}")
        
        return output_path


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python gitlab_review_analyzer_llm.py <config_file>")
        print("\nExample: python gitlab_review_analyzer_llm.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    analyzer = GitLabReviewAnalyzer(config_path)
    analyzer.run()


if __name__ == "__main__":
    main()
