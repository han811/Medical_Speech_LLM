import connector
import language
import speech
from .base import BaseVoiceLLM
from .connector import BaseConnector
from .language import BaseLLM
from .speech import BaseSpeechEncoder


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

    def load_model(self):
        self.connector = connector.make(self.connector_name)()
        self.speech_encoder = speech.make(self.speech_encoder_name)()
        self.llm_model = language.make(self.llm_model_name)()

    def forward(self, x):
        raise ValueError("Currently not implemented")
