# 🔌 LLM Lab: OpenAI Agents + MCP Integration (Local Dev)

## Introduction

In this lesson, we’ll learn how to integrate OpenAI’s Agents SDK with the Model Context Protocol (MCP) for a powerful local development setup using Azure OpenAI. MCP is an **open protocol** that standardizes how AI models interact with external tools and data, acting like a “USB-C for AI” to plug in new capabilities. By combining Azure OpenAI LLM with MCP, we can enable an AI agent to **securely use local and remote tools** through a standardized interface. This transforms our AI from an isolated chatbot into a context-aware assistant that can leverage files, databases, APIs, and even other AI function.

We will cover:

- **Agents as MCP Clients (Hosts):** Using OpenAI’s Agents SDK to have an AI agent call MCP tools.
- **Building MCP Servers:** Exposing custom functions (including those powered by Azure OpenAI) as MCP-compliant servers that other agents/clients can call.
- **Local Dev Environment Setup:** Setting up FastAPI, Azure OpenAI, and the Agents SDK to run an agent and MCP server on your machine.
- **Hands-on Assignment:** Creating an MCP server that extracts structured data from unstructured resume text, and an OpenAI agent that calls this server to decide “hire” or “not hire.”
- **Agents vs MCP vs A2A:** Clarifying when to use the OpenAI Agents SDK, when to use MCP for tools, and when Agent-to-Agent (A2A) orchestration is relevant in larger systems.
- **Security Warnings:** Important tips on running MCP servers safely in a local setup – use only verified tools and always inspect server code before running.

By the end, you’ll know how to plug Azure OpenAI into a local tool ecosystem using MCP, turning your development machine into a mini sandbox for AI + tools integration. Let’s dive in! 🚀

## OpenAI Agents as MCP Clients (Hosts)

OpenAI’s Agents SDK allows you to create an AI *agent* (backed by an LLM like GPT-4 or Azure OpenAI’s models) that can use tools to fulfill tasks. With MCP, these tools are provided by **MCP servers**, and the agent acts as an MCP **host/client** that can dynamically discover and invoke any tool that a compliant MCP server offers – all through a standard protocol.

**How it works:** The OpenAI Agents SDK has built-in support for MCP, so you can attach MCP servers to your agent. Under the hood, the agent will query each attached server for its available tools (by calling `list_tools()`), making the LLM aware of those tool functions. When the LLM decides to use one of the tools, the SDK routes the request to the appropriate MCP server (calling `call_tool()` on that server. This flow is handled automatically, letting you focus on what the tools do rather than how to call them.

> [NOTE!]
> Before Agents SDK, ones have to use function calling and openai-to-mcp bridge for interaction over MCP protocol.

There are two kinds of MCP server connections the SDK supports:

- **STDIO Servers (Local subprocess):** The server runs as a subprocess on your machine and communicates via [standard I/O pipes](https://openai.github.io/openai-agents-python/mcp/#:~:text=Currently%2C%20the%20MCP%20spec%20defines,the%20transport%20mechanism%20they%20use). Use this for local tools packaged as executables or scripts. The SDK provides `MCPServerStdio` to launch such a server.
- **SSE Servers (HTTP Server-Sent Events):** The server runs as a web service (could be local or remote) and communicates over [HTTP using Server-Sent Events](https://openai.github.io/openai-agents-python/mcp/#:~:text=Currently%2C%20the%20MCP%20spec%20defines,the%20transport%20mechanism%20they%20use). Use `MCPServerSse` to connect to these by URL. This is great for services running in FastAPI or similar frameworks, or any remote MCP tool service.

**Attaching an MCP server to an agent:** You can add one or multiple MCP servers when instantiating your agent. For example:

```python
from openai_agents import Agent, MCPServerSse

# Suppose we have an MCP server running locally on port 8000
tools_server = MCPServerSse(url="http://localhost:8000") 

agent = Agent(
    name="Assistant",
    model="gpt4o",  # Azure OpenAI deployment name
    instructions="You are a helpful assistant that can use tools.",
    mcp_servers=[tools_server]
)
```

In this snippet, `tools_server` is an MCP server client that knows how to reach our server (here via HTTP on localhost). By passing `mcp_servers=[tools_server]` into the Agent, we instruct the SDK to [load that server’s tool list](https://openai.github.io/openai-agents-python/mcp/#:~:text=Using%20MCP%20servers) on each run. The LLM will see those tools (with their names, descriptions, and schemas) in its system/context, and can decide to invoke them. When it does, the Agents SDK will forward the call to our MCP server and retrieve the result, which the LLM can then use.

**Example:** If the MCP server provides a tool called `search_files(query)`, the agent might decide to call `search_files("project timeline")`. The SDK sends this request to the server; the server executes the search and returns results; the agent then uses those results to formulate its final answer. All of this happens through the standard MCP interface – no custom integration code for the specific tool is needed in the agent logic, see [Getting Started with MCP using OpenAI Agents | Dave Davies](https://www.linkedin.com/posts/daverdavies_getting-started-with-mcp-using-openai-agents-activity-7311147410360020993-xmsB#:~:text=OpenAI%20%E2%80%99s%20Agents%20SDK%20now,the%20full%20tutorial%20%2020). This standardized approach **"provides structured access to 🛠️ tools, files, and prompts with zero custom integration"**, meaning once your agent speaks MCP, it can plug into any MCP-compliant tool easily.

**MCP in IDEs:** It’s worth noting that *other applications* (like IDEs or chat UIs) can also act as MCP hosts. For example, VS Code or Chainlit can maintain an MCP client session to [use these same servers](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=MCP%20operates%20on%20a%20client,architecture). The benefit of MCP is that tools become reusable across different AI agents and interfaces. In our case, we focus on the OpenAI Agents SDK as the host within a Python environment, but the concept generalizes widely.

🤖 **Pro Tip:** The official OpenAI Agents documentation provides end-to-end examples of using MCP servers with agents. If you have the Agents SDK installed, check out the `examples/mcp` in the repository for a hands-on demonstration [Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=End). This can help you see how the agent prompt is constructed to include the MCP tools, and how the tool outputs appear in the agent’s reasoning.

## Exposing Functions as MCP Servers (Tool Providers)

Now let’s look at the other side of the equation: creating an MCP **server**. An MCP server is simply a program that exposes one or more “tools” (functions or actions) in a standardized way so that any MCP client can discover and call  ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=,based%20tools))100】. Tools can range from simple file operations and database queries to complex actions like running an ML model. In our case, we’re particularly interested in exposing **OpenAI-powered functions** as tools – meaning the server’s implementation of a tool might itself call an OpenAI (Azure) model to get its job done.

**How MCP servers work:** According to the MCP spec, a server needs to do a few things: advertise its available tools (their names, descriptions, and input/output schema) and handle incoming requests to execute those tools, returning results. The communication with clients follows the MCP protocol (which is built on JSON over STDIO or JSON over SSE streams, using a pattern similar to JSON-RPC  ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=with%20external%20data%20and%20tools,With%20diverse%20use%20cases%20like))L16】. Fortunately, as a developer you don’t have to craft raw JSON-RPC by hand – you can use SDKs or frameworks to help. The **Model Context Protocol Servers** repository provides reference implementations in both *TypeScript* and *Python* for various tools, demonstrating how to implement MCP endpo ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=The%20servers%20in%20this%20repository,SDK%20%20or%20%2080))272】.

There are two main approaches to building an MCP server in Python:

- **Using the Python MCP SDK:** This SDK can handle protocol details for you. You define your tool functions (with their schemas) and the SDK takes care of exposing them via STDIO or SSE. For example, the reference *Filesystem* server uses the SDK to expose file operations with configurable access cont ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=%2A%20Everything%20%20,search%2C%20and%20manipulate%20Git%20repositories))284】. Using the SDK is convenient as it ensures compliance with MCP standards. (Check the `modelcontextprotocol` PyPI package or the GitHub examples for usage.)
- **Custom HTTP Server (FastAPI):** For learning purposes, you can also implement an MCP server “manually” using a web framework like FastAPI. Essentially, you’ll create endpoints for the required MCP actions. Typically:
    - A **`/list-tools`** endpoint (or similar) that returns the list of available tools and their schemas (this is how the agent knows what functions exist).
    - A **`/call/<tool_name>`** endpoint (or a unified endpoint) that accepts a request with tool arguments, executes the tool, and streams or returns the result.

  With FastAPI, you might use its support for streaming responses (Server-Sent Events) for the call executions, as MCP over SSE expects a stream of events (which can include tool outputs or intermediate progress). This approach gives you flexibility to integrate any custom logic. You will need to follow the MCP message format expected by the client SDK – for SSE, that means sending JSON events that the client library will interpret.

**OpenAI-powered tool example:** Imagine an MCP server that offers a tool **`summarize_text(text, max_length)`**. Internally, when this tool is called, the server could invoke an Azure OpenAI completion API to summarize the given text. The server then packages the summary as the result to return to the client. From the agent’s perspective, it just sees a tool named "summarize_text" that it can call – it doesn’t care that behind the scenes GPT-4 actually did the work. This pattern is powerful: it lets you wrap advanced AI capabilities as **reusable tools**. In fact, you could chain AI this way (one agent calling a tool that calls another model).

To implement such a server with FastAPI: you would set up a POST route like `/call/summarize_text` that reads the input JSON (with fields “text” and “max_length”), calls `openai.ChatCompletion.create()` with your Azure OpenAI credentials, and returns the response (probably just the summary string). Additionally, implement a GET route `/list-tools` that returns a JSON listing something like:

```json
{
  "tools": [
    {
      "name": "summarize_text",
      "description": "Summarize a piece of text to a given length.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "max_length": {"type": "integer"}
        },
        "required": ["text"]
      }
    }
    // ...other tools...
  ]
}
``` 

The Agents SDK’s `MCPServerSse` or `MCPServerStdio` will typically handle calling these endpoints under the hood when you do `server.list_tools()` or `server.call_tool()`. In our agent code earlier, those methods are invoked automatically. As long as your server replies with the expected JSON, the agent will receive the tool info and be able to call it.

For detailed guidance, refer to the **Official MCP integration docs by Open ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=The%20Model%20context%20protocol%20,From%20the%20MCP%20docs)) ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=Using%20MCP%20servers))150】 and browse community-built examples in the **modelcontextprotocol/servers GitHub re ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=The%20servers%20in%20this%20repository,SDK%20%20or%20%2080))272】. You can find servers for everything from web search to SQL databases in that repo – for instance, a **PostgreSQL** server that lets agents run read-only que ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=%2A%20Google%20Maps%20%20,Channel%20management%20and%20messaging%20capabilities))297】, or a **Slack** server to send mess ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=%2A%20Redis%20%20,Time%20and%20timezone%20conversion%20capabilities))300】. These examples can serve as templates for creating your own MCP server. Our focus will be a custom server that uses an LLM to analyze resumes (coming up in the assignment!).

## Local Development Setup (FastAPI + Azure OpenAI + Agents SDK)

Setting up a local dev environment for MCP experimentation involves a few pieces working together. Here’s a step-by-step checklist:

1. **Azure OpenAI Configuration:** Make sure you have access to an Azure OpenAI service with a deployed model (e.g., GPT-4 or GPT-35-Turbo). Note your endpoint URL and API key, and set them as environment variables (e.g., `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`). In Python, you’ll use the `openai` library to call the Azure OpenAI API – it supports Azure endpoints by specifying the `api_base`, `api_type="azure"`, etc. Alternatively, if using the Agents SDK’s built-in integration, you might configure the Azure OpenAI model in the Agent settings or via the environment as shown in Azure’s example  ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=from%20dotenv%20import%20load_dotenv%20load_dotenv%28))L47】.

2. **Install Dependencies:** You will need the OpenAI Agents SDK and FastAPI (plus Uvicorn for running the web server). Install them via pip:
   ```bash
   pip install openai-agents-python fastapi uvicorn[standard]
   ``` 
   (If `openai-agents-python` is not the correct package name, check OpenAI’s documentation for the installation command. It might be included in the main `openai` package by the time of reading, or a separate one on GitHub.)  
   Also install `openai` Python package if not already, for calling Azure OpenAI from your server code.

3. **MCP Server Implementation:** Create a FastAPI app (say, `mcp_server.py`) that defines your tool(s). For our case, we’ll implement a resume parsing function (details in the assignment section). Include endpoints: one for listing tools (e.g., `GET /mcp/tools`) and one for executing a tool (e.g., `POST /mcp/call`). If using SSE, FastAPI allows streaming responses using `StreamingResponse`. You might structure it such that a client can POST to `/mcp/call` with a JSON payload `{"tool": "<name>", "args": { ... }}` and your server will process and yield events. For simplicity, you could also implement a synchronous call (return the result in one response) — the Agents SDK will still accept that result.

   **Development tip:** Start simple. First, get your server to return a static list of tools and perhaps echo back inputs on a call, to ensure the agent can connect. Then add the real logic. You can run the FastAPI app with:
   ```bash
   uvicorn mcp_server:app --reload --port 8000
   ``` 
   This will start your MCP server on `http://localhost:8000`.

4. **OpenAI Agent Setup:** In a separate script or Jupyter notebook, initialize your Agent with Azure OpenAI. For example:
   ```python
   import os
   import openai
   from openai_agents import Agent, MCPServerSse

   # Configure Azure OpenAI credentials
   openai.api_type = "azure"
   openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
   openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
   openai.api_version = "2023-03-15-preview"  # or the appropriate API version for your model

   # Set up MCP server client pointing to our FastAPI server
   tools_server = MCPServerSse(url="http://localhost:8000/mcp")

   agent = Agent(
       name="HR Assistant",
       model="gpt-4",  # might be configured internally to use Azure via openai library
       instructions="You are an HR assistant that analyzes resumes. You have a tool to parse resumes for details.",
       mcp_servers=[tools_server]
   )
   ``` 
   Note: The exact `model` parameter usage for Azure may differ; some SDKs let you pass an AzureOpenAI instance or deployment name. The key is that the agent will ultimately use Azure OpenAI under the hood for language generation, and the tool will be accessible via `tools_server`.

5. **Test the Connection:** Run a simple prompt through the agent to verify it sees the tool. For example:
   ```python
   response = agent.run("List the tools you have and describe them.")
   print(response)
   ``` 
   The agent should ideally respond with the tool name and description (because it sees that from the MCP server’s `list_tools`). If it does, congratulations – your agent and MCP server are talking! 🎉 From here, you can prompt the agent to use the tool on some input (we will do exactly that in the assignment).

Throughout this process, use the **official MCP documentat ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=The%20Model%20context%20protocol%20,From%20the%20MCP%20docs))L115】 and the **Azure OpenAI + MCP integration guide** (Microsoft’s blog  ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=MCP%20has%20the%20potential%20to,Ultimately%2C%20MCP%20aims%20to)) ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=,based%20tools))L100】 as references. The Azure blog demonstrates using Chainlit (a UI) plus OpenAI’s Python library to attach MCP servers to a chat interface, which parallels what we’re doing in code. They flatten the tool list and feed it into the Azure OpenAI chat completion as func ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=Next%20thing%20we%20need%20to,chat%20session%20after%20flattening%20it)) ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=mcp_tools%20%3D%20cl.user_session.get%28,for%20tool%20in%20tools))L151】 – the Agents SDK essentially automates that flattening and calling for you. Understanding both perspectives (manual integration vs. SDK) can deepen your insight.

## Assignment: AI-Powered Resume Screener (MCP Server + Agent)

It’s time to apply what we’ve learned. In this assignment, you will build a mini application that helps decide whether to hire a job candidate based on their resume, by combining an MCP server tool and an OpenAI agent. The scenario is: we have unstructured resume text, and we want our AI agent to analyze it and return a structured **“Hire” or “No Hire” decision** (with reasoning). We’ll do this in two parts:

**Part A – MCP Resume Parser Server:** Develop an MCP server (using FastAPI or the Python MCP SDK) that exposes a tool, e.g., `parse_resume(resume_text)`. When this tool is called, the server should extract structured information from the free-form resume text. At a minimum, have it identify the candidate’s **years of experience**, **key skills**, and maybe **current or last job title**. You can choose what details to extract, but ensure the output is a structured format (e.g., a JSON with fields like `{"years_experience": 5, "skills": ["Python", "SQL", ...], "last_position": "Senior Data Scientist"}`).
Make sure your FastAPI server returns the extracted data as the result of the tool call. Also decide on the input schema: likely just a single `resume_text` string input. Document the tool’s name, description, and inputs in the server’s tool list response.

**Part B – Hiring Decision Agent:** Using the OpenAI Agents SDK, create an agent that can use the above tool to make a hiring recommendation. This agent’s prompt (system message) should instruct it to evaluate resumes objectively and call the `parse_resume` tool when given a raw resume. The agent’s logic will be:
1. Call `parse_resume` to get structured info.
2. Analyze the parsed info (e.g., check if required skills match, sufficient experience, etc.). You can bake some simple criteria into the instructions, for example: “If years_experience < 2, likely no hire. If key skills include X and Y with over 5 years experience, strong hire,” etc., or let the model reason it out.
3. Respond with a decision "Hire" or "No Hire" and a brief explanation.

You might need to format the agent’s answer in a consistent way (perhaps as "**Decision:** Hire (or No Hire)\n**Reason:** ..."). The main goal is to see the agent *autonomously* using the MCP tool to improve its decision-making. This is where the magic of tool-use comes in: without the tool, the agent might try to parse the raw text on its own (which it could, but here we simulate a specialized parser doing it). With the tool, we inject a structured analysis step that can make the agent’s job easier and more reliable.

**Steps to follow:**

1. **Implement and Run the Server:** Code the resume parser FastAPI app. Test its `/parse_resume` internally by sending a sample resume and checking the output. Run the server (e.g. on localhost:8000).
2. **Set Up the Agent:** Instantiate your agent with Azure OpenAI (GPT-4 or GPT-35) and attach the MCP server (as shown earlier). Craft the system message to clearly instruct the agent on its task and tool usage. For example: *“You are an HR assistant. You will be given a candidate’s resume as text. Use the `parse_resume` tool to extract details from the resume, then decide whether to hire the candidate. Respond with 'Hire' or 'No Hire' and explain your reasoning based on the extracted info.”* The agent will then receive the user prompt containing the raw resume text.
3. **Test with Sample Resumes:** Provide a couple of resume inputs (they can be short paragraphs describing a fictional candidate’s background). Observe the agent’s behavior. It should call the tool (you’ll see this in logs or the agent’s chain-of-thought if you enable logging). Once the tool returns data, the agent should output a decision. Verify if the decision seems reasonable given the criteria. For instance, try one resume that clearly meets criteria (strong experience, relevant skills) and one that does not – see if the agent differentiates them.
4. **Iterate:** You may need to tweak the parsing details or the agent prompt to get better results. For example, if the agent ignores the tool, ensure the instructions emphasize using it. If the tool’s output is incomplete, refine the parsing prompt to Azure OpenAI. This is an exercise in orchestrating an AI workflow, so some prompt engineering and debugging is normal.

Finally, prepare to **demo your solution**. You should be able to run the agent on a sample input and show the tool invocation and final answer. This assignment will solidify your understanding of how MCP can integrate a custom tool into an AI’s reasoning process in real-time. It’s a glimpse of building your own “Copilot” that can consult specialized skills on demand!

## OpenAI Agents SDK vs MCP vs A2A – When to Use What

With all these terms and tools, it’s important to understand their roles in your AI development toolbox:

- **OpenAI Agents SDK (Single-Agent Orchestration):** Use the Agents SDK when you want to *enable an LLM to use tools or functions*. It’s perfect for building an AI agent that can take a user query and decide to call functions (tools) to help answer that query. The SDK handles the heavy lifting of prompting the model with available tools, parsing function call outputs, and so on. In our case, we used it to connect one agent to our resume parsing tool. If you’re building an application where one AI assistant needs to interact with your local code or external APIs, the Agents SDK is your go-to. It’s focused on the single-agent scenario: **one LLM agent, many tools**.

- **Model Context Protocol (MCP – Standardized Tools):** MCP comes into play when you want a **standard interface for tools** that can be reused across different agents and environments. Think of MCP as a *plugin system*. If you only care about one specific AI agent in one codebase, you might not need MCP – you could directly use function calling or custom integration. But if you envision using tools in multiple places (maybe today in a Python script, tomorrow in VS Code, next week in a web app), MCP saves you from reinventing the wheel each time. You implement the tool *once* as an MCP server, and any compliant agent (OpenAI’s, Anthropic’s, Microsoft’s, etc.) can u ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=scalable%20data%20exchange%20between%20LLMs%2FAI,Key%20elements%20from%20the%20Model))-L72】. In short, use MCP when you want **portability and interoperability** for your AI tools. Our resume parser could, for example, be plugged into a VS Code extension that supports MCP, enabling a developer to get AI-driven resume analysis in their editor – without us writing new code for that environment. MCP is also useful in team settings: everyone can share a library of MCP servers (for databases, internal APIs, etc.), knowing they will work with any agent that speaks MCP.

- **Agent-to-Agent (A2A) Orchestration:** Sometimes one agent isn’t enough. **A2A** refers to orchestrating multiple AI agents together (agent-to-agent communication or coordination). This is relevant in larger systems or complex tasks that can be broken down. For example, one agent could be a “Researcher” that gathers information, and another a “Writer” that produces a report; or a supervisor agent might break a job into subtasks for specialist sub-agents. The OpenAI Agents SDK provides a concept called **handoffs** (or delegations) for this purpose, allowing an agent to delegate tasks to another  ([Orchestrating multiple agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/multi_agent/#:~:text=,planning%2C%20report%20writing%20and%20more))L124】. Orchestration can be done *via the LLM* (one agent figures out when to invoke another) or *via code* (you explicitly sequence ag ([Orchestrating multiple agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/multi_agent/#:~:text=Orchestrating%20via%20code))L148】. Use A2A when your solution would benefit from specialized experts or parallel processes. However, be mindful: multi-agent systems add complexity and cost. They shine in scenarios where tasks are too much for one agent to handle well in a single prompt. In our bootcamp, A2A might not be heavily used, but it’s good to know the concept. **Summary:** Agents SDK (single agent + tools) is usually the first step. If your tool ecosystem grows, formalize it with MCP for reuse. If your problem grows, consider multiple agents coordinating (A2A), which the Agents SDK can facilitate through handoffs or orchestrator code.

## Security Considerations for MCP in Local Setup 🔒

Whenever you integrate external tools or run servers locally, **security must be top-of-mind**. Here are some important guidelines and warnings when working with MCP servers and agents:

- **Run Trusted Servers Only:** MCP servers, by design, can have extensive access – a filesystem server can read/write files, a database server can query your DB, etc. **Do NOT run MCP servers from unverified sources** on your machine without inspecting the code. Since many community MCP servers are open-source, review the code or at least understand what commands it will execute. For example, an innocent-sounding server could potentially contain malicious code that, when run, could steal data or damage files. Only use official or well-reviewed community servers. When in doubt, sandbox the server (e.g., run in a Docker container with limited volume mounts).

- **Principle of Least Privilege:** Even for your own MCP servers, limit their access to only what’s necessary. For instance, if you create a filesystem tool for the agent, consider restricting it to a specific directory (the official Filesystem server supports configurable access controls for this  ([GitHub - modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers#:~:text=%2A%20Everything%20%20,search%2C%20and%20manipulate%20Git%20repositories))-L284】). If you make a database tool, use read-only credentials if the task doesn’t need writes. This way, even if the agent misuses a tool or a bug is exploited, the damage is contained.

- **Protect Secrets:** Your MCP server might need credentials (like API keys for external services or your Azure OpenAI key if it calls the API). Avoid hard-coding these in code that could be shared. Use environment variables or secure config files. Also, be cautious about what the agent can output – if a tool returns sensitive info (say, contents of a private file), make sure your agent prompt doesn’t simply regurgitate it to the user without filters.

- **MCP Server Authentication:** In local dev, you might run the server open on `localhost`. That’s fine, but if you ever expose it beyond your machine, implement auth (e.g., API keys or OAuth). MCP itself is transport-agnostic, so you’d have to secure the channel (HTTPS, tokens) as appropriate. The Agents SDK will likely allow setting headers or auth for `MCPServerSse` connections; consult its docs if needed. For now, in a closed local environment, this is less of an issue, but still worth noting.

- **Stay Updated:** The MCP spec and Agents SDK are evolving. New security features (like more granular permissions or sandboxing options) could appear. Keep your dependencies up-to-date, and read release notes. For instance, caching of tool lists is available to reduce l ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=Caching))-L163】, but be mindful if your tool list might change or if a stale list could present wrong capabilities. Always invalidate or refresh caches when needed to avoid security or consistency  ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=Every%20time%20an%20Agent%20runs%2C,tool%20list%20will%20not%20change))-L165】.

- **Debugging and Logs:** During development, log what your MCP server is doing (e.g., print when a request comes in and what it’s doing with it). The Agents SDK may also provide traces of tool ([Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/#:~:text=View%20complete%20working%20examples%20at,examples%2Fmcp))-L177】. Monitoring these logs helps detect any odd or unintended tool usage. If your agent tries to do something out of scope, you’ll catch it. This is part of AI safety – ensuring the agent + tool combo behaves as expected.

**⚠️ Caution:** Always remember that an AI agent with powerful tools is essentially running code on your behalf. You wouldn’t execute arbitrary code from strangers on your laptop; similarly, constrain and supervise what the agent can do with tools. OpenAI’s documentation reminds that with great power (tools), comes great responsibility in design and ove ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=MCP%20has%20the%20potential%20to,Ultimately%2C%20MCP%20aims%20to)) ([Model Context Protocol (MCP): Integrating Azure OpenAI for Enhanced Tool Integration and Prompting | Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/model-context-protocol-mcp-integrating-azure-openai-for-enhanced-tool-integratio/4393788#:~:text=,based%20tools))-L100】. Treat your MCP servers as an extension of your application’s attack surface.

---  

❤️ **LLM LAB – 2025**
