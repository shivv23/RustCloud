#![allow(dead_code)]
use async_trait::async_trait;

use crate::errors::CloudError;
use crate::traits::llm_provider::LlmProvider;
use crate::types::llm::{EmbedResponse, LlmRequest, LlmResponse, ToolCallResponse, ToolDefinition};

pub enum LlmProviderType {
    GoogleVertex(google_vertexai::GoogleVertexAI),
    AmazonBedrock(amazon_bedrock::AmazonBedrock),
    AzureOpenAI(azure_openai::AzureOpenAI),
}

impl LlmProviderType {
    pub fn google_vertex(project_id: String, location: Option<String>) -> Self {
        LlmProviderType::GoogleVertex(google_vertexai::GoogleVertexAI::new(project_id, location))
    }

    pub fn amazon_bedrock(region: String) -> Self {
        LlmProviderType::AmazonBedrock(amazon_bedrock::AmazonBedrock::new(region))
    }

    pub fn azure_openai(endpoint: String, api_key: String) -> Self {
        LlmProviderType::AzureOpenAI(azure_openai::AzureOpenAI::new(endpoint, api_key))
    }

    pub fn name(&self) -> &'static str {
        match self {
            LlmProviderType::GoogleVertex(_) => "Google Vertex AI",
            LlmProviderType::AmazonBedrock(_) => "Amazon Bedrock",
            LlmProviderType::AzureOpenAI(_) => "Azure OpenAI",
        }
    }
}

mod google_vertexai {
    pub use crate::gcp::gcp_apis::artificial_intelligence::vertex::GoogleVertexAI;
}

mod amazon_bedrock {
    pub use crate::aws::aws_apis::artificial_intelligence::bedrock::AmazonBedrock;
}

mod azure_openai {
    pub use crate::azure::azure_apis::ai::openai::AzureOpenAI;
}

#[async_trait]
impl LlmProvider for LlmProviderType {
    async fn generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        match self {
            LlmProviderType::GoogleVertex(client) => client.generate(req).await,
            LlmProviderType::AmazonBedrock(client) => client.generate(req).await,
            LlmProviderType::AzureOpenAI(client) => client.generate(req).await,
        }
    }

    async fn stream(
        &self,
        req: LlmRequest,
    ) -> Result<crate::traits::llm_provider::LlmStream, CloudError> {
        match self {
            LlmProviderType::GoogleVertex(client) => client.stream(req).await,
            LlmProviderType::AmazonBedrock(client) => client.stream(req).await,
            LlmProviderType::AzureOpenAI(client) => client.stream(req).await,
        }
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        match self {
            LlmProviderType::GoogleVertex(client) => client.embed(texts).await,
            LlmProviderType::AmazonBedrock(client) => client.embed(texts).await,
            LlmProviderType::AzureOpenAI(client) => client.embed(texts).await,
        }
    }

    async fn generate_with_tools(
        &self,
        req: LlmRequest,
        tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        match self {
            LlmProviderType::GoogleVertex(client) => client.generate_with_tools(req, tools).await,
            LlmProviderType::AmazonBedrock(client) => client.generate_with_tools(req, tools).await,
            LlmProviderType::AzureOpenAI(client) => client.generate_with_tools(req, tools).await,
        }
    }
}