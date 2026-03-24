#![allow(clippy::too_many_arguments, clippy::new_without_default, dead_code)]
use async_trait::async_trait;

use crate::errors::CloudError;
use crate::traits::llm_provider::LlmProvider;
use crate::types::llm::{
    EmbedResponse, FinishReason, LlmRequest, LlmResponse, ModelRef, ToolCallResponse,
    ToolDefinition, UsageStats,
};

#[derive(Debug)]
pub struct AmazonBedrock {
    region: String,
}

impl AmazonBedrock {
    pub fn new(region: String) -> Self {
        Self { region }
    }

    pub async fn from_env() -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let region = config.region().map(|r| r.to_string()).unwrap_or_default();
        Self { region }
    }

    fn resolve_model_id(&self, model: &ModelRef) -> String {
        match model {
            ModelRef::Provider(id) => id.clone(),
            ModelRef::Logical { family, tier } => {
                let suffix = tier.as_deref().unwrap_or("001");
                format!("{}:{}", family, suffix)
            }
            ModelRef::Deployment(dep) => dep.clone(),
        }
    }
}

#[async_trait]
impl LlmProvider for AmazonBedrock {
    async fn generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        let _model_id = self.resolve_model_id(&req.model);
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect();

        let content = messages
            .first()
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("Bedrock integration placeholder")
            .to_string();

        Ok(LlmResponse {
            text: format!("[AWS Bedrock] Received: {}", content),
            finish_reason: FinishReason::Stop,
            usage: Some(UsageStats {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        })
    }

    async fn stream(
        &self,
        _req: LlmRequest,
    ) -> Result<crate::traits::llm_provider::LlmStream, CloudError> {
        Err(CloudError::Unsupported {
            feature: "streaming not yet implemented for Bedrock",
        })
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        Err(CloudError::Unsupported {
            feature: "embeddings not yet implemented for Bedrock",
        })
    }

    async fn generate_with_tools(
        &self,
        _req: LlmRequest,
        _tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        Err(CloudError::Unsupported {
            feature: "tools not yet implemented for Bedrock",
        })
    }
}