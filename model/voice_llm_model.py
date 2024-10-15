import torch
import gin

from . import connector
from . import language
from . import speech
from .base import BaseVoiceLLM
from .connector import BaseConnector
from .language import BaseLLM
from .speech import BaseSpeechEncoder
import torchaudio


@gin.configurable
class VoiceLLM(BaseVoiceLLM):
    def __init__(
        self,
        connector_name: str,
        speech_encoder_name: str,
        llm_model_name: str,
    ):
        super().__init__(
            connector_name,
            speech_encoder_name,
            llm_model_name,
        )

        self.connector: BaseConnector = None
        self.speech_encoder: BaseSpeechEncoder = None
        self.llm_model: BaseLLM = None

        self.load_model()

    def load_model(self):
        self.connector = connector.make(self.connector_name)()
        self.speech_encoder = speech.make(self.speech_encoder_name)()
        self.llm_model = language.make(self.llm_model_name)()

    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out

    def encode(
        self,
        mel,
        pre_tokenized_ids,
        post_tokenized_ids,
        output_tokenized_ids,
    ):
        batch_size = mel.shape[0]

        mel = mel.to("cuda")

        speech_embeds = self.speech_encoder(mel)
        speech_embeds = self.connector(speech_embeds)

        embedder = self.llm_model.model.model.embed_tokens
        # embedder = self.llm_model.model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat(
            [
                pre_prompt_embeds,
                speech_embeds,
                post_prompt_embeds,
                output_prompt_embeds,
            ],
            dim=1,
        )
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(
            combined_embeds.device
        )

        input_token_length = (
            pre_tokenized_ids.shape[1]
            + speech_embeds.shape[1]
            + post_tokenized_ids.shape[1]
        )
        label_ids = (
            torch.cat(
                [
                    torch.ones(
                        [batch_size, input_token_length], device=combined_embeds.device
                    )
                    * -100,
                    output_tokenized_ids,
                ],
                1,
            )
            .to(combined_embeds.device)
            .to(torch.int64)
        )
        return combined_embeds, atts, label_ids

    def get_preprocesor(self):
        return self.speech_encoder.preprocessor, self.llm_model.preprocessor

    # def preprocess(self, batch):
    #     speech = list()
    #     for audio_path in batch["audio_path"]:
    #         waveform, sample_rate = torchaudio.load(audio_path)
    #         speech.append(waveform)
    #     speech = torch.concat(speech)

    #     pre_words = batch["question"]
    #     post_words = "<post token>"
    #     output_words = batch["answer"]

    #     speech_features = self.speech_encoder.preprocessor(speech)
    #     pre_tokenized_ids = self.llm_model.preprocessor(pre_words)
    #     post_tokenized_ids = self.llm_model.preprocessor(post_words)
    #     output_tokenized_ids = self.llm_model.preprocessor(output_words)

    #     return (
    #         speech_features,
    #         pre_tokenized_ids,
    #         post_tokenized_ids,
    #         output_tokenized_ids,
    #     )
