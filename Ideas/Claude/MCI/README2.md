# Markdown Context Injector for LLMs

A novel approach to context injection using pure Markdown syntax to structure, prioritize, and inject contextual information into LLM prompts.

## Core Concept

This system uses Markdown’s hierarchical structure, emphasis markers, and semantic elements to create a “context stack” that LLMs naturally understand. The key insight is that LLMs are trained to respect Markdown formatting, making it an ideal medium for meta-instructions.

-----

## The Context Injection Format

### Level 1: Critical Context (H1 + Bold + Blockquotes)

```markdown
# ⚡ PRIMARY DIRECTIVE

> **CRITICAL**: This context must be preserved throughout the conversation
> 
> **Priority Level**: MAXIMUM
> **Scope**: Global
```

### Level 2: High Priority Context (H2 + Bold)

```markdown
## 🎯 Core Context

**Domain**: Software Engineering
**User Level**: Senior Developer
**Current Task**: Code Review
```

### Level 3: Standard Context (H3 + Lists)

```markdown
### 📋 Session Context

- **Conversation ID**: conv_12345
- **Start Time**: 2026-01-13T10:30:00Z
- **Previous Topics**: API design, database optimization
- **User Preferences**: 
  - Concise explanations
  - Code examples preferred
  - Avoid excessive jargon
```

### Level 4: Background Context (H4 + Collapsed Details)

```markdown
#### 📚 Background Knowledge

<details>
<summary>Project Context (Click to expand)</summary>

- Project: E-commerce Platform Redesign
- Tech Stack: Node.js, React, PostgreSQL
- Team Size: 5 developers
- Timeline: 3 months remaining
</details>
```

-----

## Advanced Features

### 1. Priority Weighting System

Use emoji + formatting combinations to signal importance:

```markdown
# 🔴 CRITICAL (Weight: 100)
## 🟠 HIGH (Weight: 80)
### 🟡 MEDIUM (Weight: 60)
#### 🟢 LOW (Weight: 40)
##### ⚪ BACKGROUND (Weight: 20)
```

### 2. Temporal Context Windows

```markdown
## ⏰ Temporal Context

| Time Frame | Relevance | Context |
|------------|-----------|---------|
| **NOW** (Last 5 min) | 🔴 100% | User asked about authentication bugs |
| **RECENT** (Last hour) | 🟠 75% | Discussed JWT implementation |
| **SESSION** (Today) | 🟡 50% | Reviewed database schema |
| **HISTORICAL** (This week) | 🟢 25% | Discussed architecture patterns |
```

### 3. Dynamic Context Injection

Use horizontal rules to create “context layers”:

```markdown
---
**[CONTEXT LAYER 1: IMMEDIATE]**
User just mentioned: "The login endpoint is returning 401"
---
**[CONTEXT LAYER 2: CONVERSATION]**
We've been debugging authentication for 10 minutes
---
**[CONTEXT LAYER 3: PROJECT]**
This is a microservices architecture with OAuth2
---
```

### 4. Scoped Context Blocks

```markdown
### 🎯 Scoped Contexts

#### For Next Response Only:
> **TEMP_CONTEXT**: User is frustrated, be empathetic
> **EXPIRES**: After 1 response

#### For This Subtopic:
> **TOPIC_CONTEXT**: Focus on security best practices
> **EXPIRES**: When topic changes

#### For This Session:
> **SESSION_CONTEXT**: User prefers TypeScript examples
> **EXPIRES**: End of conversation
```

### 5. Conditional Context Activation

```markdown
### ⚙️ Conditional Contexts

**IF** user mentions "performance" **THEN** activate:
- Database query optimization context
- Caching strategy context
- Load testing experience context

**IF** user shows confusion **THEN** activate:
- Simplify explanations mode
- Provide more examples
- Check for understanding
```

-----

## Complete Example: Code Review Session

```markdown
# ⚡ ACTIVE SESSION CONTEXT

> **Role**: Senior Code Reviewer
> **User**: Mid-level Backend Developer
> **Task**: Reviewing Pull Request #342
> **Tone**: Constructive and educational

---

## 🎯 Immediate Context

**Current File**: `src/services/AuthService.ts`
**Lines of Interest**: 45-67
**Issue Detected**: Potential SQL injection vulnerability
**User Mood**: Receptive to feedback

---

### 📋 PR Context

| Attribute | Value |
|-----------|-------|
| PR Number | #342 |
| Author | @johndoe |
| Title | "Add user authentication endpoints" |
| Changes | +287, -43 |
| Files Changed | 8 |
| Status | Awaiting Review |

---

### 🧠 Knowledge Context

<details>
<summary>User's Known Experience</summary>

**Strong in:**
- Node.js fundamentals
- RESTful API design
- Unit testing

**Learning:**
- Security best practices
- TypeScript advanced types
- Database optimization

**Gaps:**
- OWASP Top 10
- JWT implementation details
- Rate limiting strategies
</details>

---

### 📚 Project Context

<details>
<summary>Codebase Standards</summary>

- **Linting**: ESLint with Airbnb config
- **Testing**: Jest, 80% coverage minimum
- **Security**: All inputs must be validated
- **Documentation**: JSDoc required for public APIs
- **Error Handling**: Custom error classes, no bare throws
</details>

---

### ⏰ Conversation History (Last 3 Turns)

1. **User**: "I've implemented the login endpoint"
2. **Assistant**: "Let me review the authentication flow..."
3. **User**: "Is this parameterized query safe?"

---

### 🎯 Next Response Guidelines

> **Primary Goal**: Explain SQL injection vulnerability
> **Secondary Goal**: Teach parameterized queries
> **Approach**: Show vulnerable code → Explain risk → Demonstrate fix
> **Tone**: Encouraging but clear about security importance
> **Length**: Medium (2-3 paragraphs + code example)

---

**[END CONTEXT BLOCK]**
```

-----

## Implementation Strategy

### Method 1: Prepend Context to User Messages

```markdown
<!-- INJECTED CONTEXT START -->
# 🎯 Active Context for This Query
- User: Senior Developer
- Topic: Authentication
- Mood: Problem-solving mode
<!-- INJECTED CONTEXT END -->

[User's actual message here]
```

### Method 2: Hidden Comment Blocks

```markdown
<!--
CONTEXT_INJECTION:
{
  priority: "high",
  user_role: "developer",
  session_duration: "15min",
  topics_discussed: ["API", "security"],
  next_response_tone: "technical"
}
-->

[User's message]
```

### Method 3: Footnote-Style Context

```markdown
User's message here[^context]

[^context]: 
    **Session Context**:
    - Duration: 15 minutes
    - Topics: Authentication, API Security
    - User Expertise: Senior Level
    - Preferred Style: Code-focused
```

-----

## Token Efficiency Optimization

### Compact Priority Encoding

Instead of:

```markdown
### High Priority Context
**User Type**: Developer
**Experience**: Senior
**Topic**: Security
```

Use:

```markdown
### 🎯 `DEV/SR/SEC`
```

### Hierarchical Compression

```markdown
# CTX
## U:`dev:sr` T:`auth` M:`debug` P:`urgent`
### HIST:`jwt→oauth→err401`
```

Decoded:

- User: Senior Developer
- Topic: Authentication
- Mode: Debugging
- Priority: Urgent
- History: JWT discussed, then OAuth, currently on 401 error

-----

## Context Injection API (Markdown Functions)

### Function 1: ADD_CONTEXT

```markdown
**[ADD_CONTEXT]**
- **Key**: user_preference
- **Value**: concise_answers
- **Scope**: session
- **Priority**: medium
```

### Function 2: UPDATE_CONTEXT

```markdown
**[UPDATE_CONTEXT]**
- **Key**: current_topic
- **Old**: database_design
- **New**: api_security
- **Timestamp**: 2026-01-13T10:45:00Z
```

### Function 3: REMOVE_CONTEXT

```markdown
**[REMOVE_CONTEXT]**
- **Key**: temp_debugging_mode
- **Reason**: Issue resolved
```

### Function 4: QUERY_CONTEXT

```markdown
**[QUERY_CONTEXT]**
- **What**: user_expertise_level
- **Return**: senior_developer
```

-----

## Multi-Modal Context Injection

### For Code Context

```markdown
### 💻 Code Context

```typescript
// CONTEXT: This is the current implementation
async function authenticate(username: string, password: string) {
  // User is asking about line 3 below ⬇️
  const query = `SELECT * FROM users WHERE username = '${username}'`;
  // ⬆️ This is the problematic line
}
```
```

### For Visual Context

```markdown
### 🖼️ Visual Context

**Diagram Reference**: 
```

[Client] –HTTP–> [API Gateway] –gRPC–> [Auth Service]
⬆️
User asking about this connection

```
**

### For Data Context

```markdown
### 📊 Data Context

| Variable | Current Value | Expected | Status |
|----------|---------------|----------|--------|
| user_id | `null` | `integer` | ❌ ERROR |
| token | `undefined` | `string` | ❌ ERROR |
| role | `"admin"` | `"user"` | ⚠️ SUSPICIOUS |
```

-----

## Best Practices

### DO ✅

1. **Use semantic hierarchy**: H1 for critical, H2 for important, H3 for supporting
1. **Leverage visual markers**: Emojis help LLMs parse priority quickly
1. **Keep context fresh**: Update temporal markers regularly
1. **Compress when possible**: Use abbreviations for frequent contexts
1. **Scope appropriately**: Mark contexts with expiration conditions

### DON’T ❌

1. **Over-inject**: Don’t add context that doesn’t affect the response
1. **Flatten hierarchy**: Maintain clear priority levels
1. **Use ambiguous markers**: Be explicit about what each context means
1. **Ignore token costs**: Compress contexts for long conversations
1. **Forget to clean up**: Remove expired or irrelevant context

-----

## Real-World Usage Example

```markdown
# 🎯 Live Context Injection Example

## Before Injection (Raw User Query):
"Why is my API slow?"

## After Context Injection:

---
**[CONTEXT: ACTIVE SESSION]**
# ⚡ Session State
> User: @alice (Backend Developer, 3 years exp)
> Project: E-commerce API v2
> Current Sprint: Performance optimization
> Recent Topic: Database queries taking 2-3 seconds

## 🎯 Technical Context
**Stack**: Node.js 18, PostgreSQL 14, Redis 7
**Environment**: Production (AWS us-east-1)
**Scale**: 10k req/min peak
**Known Issues**: N+1 query in product listings

## 📊 Recent Metrics
- API Latency: p95 = 2.3s (target: < 500ms)
- DB Connection Pool: 98% utilization
- Cache Hit Rate: 34% (target: > 80%)

## 🧠 Conversation Memory
1. Alice mentioned adding product reviews feature last week
2. Discussed adding indexes 2 days ago
3. Just deployed to production 4 hours ago

---

**[USER QUERY]**
"Why is my API slow?"

---

**[RESPONSE DIRECTIVES]**
> **Focus**: Connect to recent deployment
> **Hypothesis**: New feature causing N+1 queries
> **Approach**: 
> 1. Acknowledge recent changes
> 2. Suggest checking query patterns
> 3. Recommend specific debugging steps
> **Tone**: Supportive problem-solving
```

-----

## Advanced: Context Inheritance Chain

```markdown
# 🧬 Context Inheritance

## Base Context (Always Active)
```markdown
**[BASE]**
- Platform: Claude
- Interface: Chat
- Capabilities: Analysis, Code, Writing
```

## ⬇️ User Context (Inherits Base)

```markdown
**[USER] extends [BASE]**
- Name: Alice
- Role: Developer
- Expertise: Senior
```

## ⬇️ Session Context (Inherits User)

```markdown
**[SESSION] extends [USER]**
- Start: 10:30 AM
- Duration: 15 min
- Topic: API Performance
```

## ⬇️ Turn Context (Inherits Session)

```markdown
**[TURN] extends [SESSION]**
- Number: 7
- Type: Follow-up question
- Emotion: Frustrated
- Urgency: High
```

-----

## Meta-Context: Context About Context

```markdown
### 🔄 Context Management Meta-Information

**Context Freshness**:
- Last Updated: 2 minutes ago
- Staleness Risk: Low
- Validation Status: ✅ Verified

**Context Completeness**:
- User Info: 100% ✅
- Technical Details: 85% ⚠️ (missing: server logs)
- Business Context: 60% ⚠️ (missing: SLA requirements)

**Context Confidence**:
- User Intent: 95% 🔴 High
- Technical Requirements: 70% 🟡 Medium
- Success Criteria: 40% 🟢 Low (ask for clarification)
```

-----

## Conclusion

This Markdown-based context injection system provides:

- **Natural Integration**: LLMs already understand Markdown
- **Human Readable**: Developers can inspect and debug contexts
- **Hierarchical**: Clear priority and scope management
- **Flexible**: Adaptable to any domain or use case
- **Token Efficient**: Compressible without losing semantics
- **Version Controllable**: Contexts can be tracked in git

The system works because it leverages the LLM’s existing training on structured documents, turning the weakness of statelessness into a strength through explicit, well-formatted context injection.
