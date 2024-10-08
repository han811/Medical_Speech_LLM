import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import random
import numpy as np


class VoiceLLMDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        mode: str = "train",
        random_state: int = 42,
        test_size: float = 0.2,
        shuffle: bool = True,
    ):
        self.total_data = pd.read_csv(csv_file)
        if shuffle:
            self.total_data = self.total_data.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        self.len = len(self.total_data)
        self.train_len = int(self.len * test_size)
        self.test_len = self.len - self.train_len

        self.mode = mode
        self.data_frame = None
        if self.mode == "train":
            self.len = self.train_len
            self.data_frame = self.total_data[: self.train_len]
        else:
            self.len = self.test_len
            self.data_frame = self.total_data[self.train_len : self.test_len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Load audio
        data_sample = self.data_frame.iloc[idx]
        audio_path = data_sample["audio_path"]
        if pd.isna(audio_path):
            raise ValueError("There is no audio data path")
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        return waveform, data_sample, sample_rate

        return waveform, labels_str, conv_history


# class InstructionalAudioDataset(AudioDataset):
#     def __init__(self, csv_file, mode="train", random_keys_prob=0.1):
#         """
#         Initialize the class with the specified CSV file, mode, and random keys probability.

#         Args:
#             csv_file (str): The path to the CSV file.
#             mode (str, optional): The mode of the operation, defaults to 'train'.
#             random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

#         Returns:
#             None
#         """
#         super().__init__(csv_file, mode, random_keys_prob)
#         self.instruction_phrases = [
#             "Provide the details about the audio",
#             "I need the following information from the audio",
#             "Tell me about the audio regarding",
#             "Extract the following details from the audio",
#             "Give me the following information about the audio",
#             "Provide details from the audio file",
#             "I need information extracted from this speech",
#             "Detail the contents of the following audio",
#             "Share insights about this speech recording",
#             "Describe the specifics captured in this audio file",
#             "Summarize the audio's key information",
#             "Convey the details embedded in this speech",
#             "Outline the main points from this audio file",
#             "Unpack the content of the following speech",
#             "Present the facts from this audio recording",
#             "Elucidate the elements within this speech",
#             "Decipher the audio file's information",
#             "Break down the details in this speech",
#             "Analyze the following audio for details",
#             "Report on the specifics of this speech file",
#             "Transcribe the key points from this audio",
#             "Explain the content of the speech recording",
#             "Interpret the information within this audio file",
#             "Catalog the details from this speech",
#             "Narrate the findings in the audio",
#             "Recount the specifics of this speech file",
#             "Review the contents of the audio",
#             "Assess the information provided by this speech",
#             "Evaluate the details in the audio file",
#             "Investigate the speech for key information",
#             "Scrutinize the audio and provide insights",
#             "Inspect the details within this speech",
#             "Examine the audio file for specific information",
#             "Survey the speech and detail your findings",
#             "Study the audio and summarize the content",
#             "Audit the speech for important details",
#             "Appraise the audio file's key points",
#             "Annotate the specifics found in the speech",
#             "Dissect the audio to find important information",
#             "Extract insights from the speech file",
#             "Unveil the details in the audio recording",
#             "Shed light on the speech's content",
#             "Clarify the specifics within the audio file",
#             "Illuminate the information in the speech",
#             "Highlight the key points of the audio",
#             "Reveal the contents captured in the speech file",
#             "Uncover the details within the audio",
#             "Delve into the speech for essential information",
#             "Probe the audio file for details",
#             "Explore the speech recording's specifics",
#             "Research the contents of the audio",
#             "Inquire into the details of the speech",
#             "Sift through the audio for key information",
#             "Dive into the speech to extract details",
#             "Investigate the nuances of the audio file",
#             "Give me the following information about the audio",
#             "Fetch information",
#             "Give me details about the audio",
#             "what does this audio say",
#             "what is in the file",
#             "give me these details",
#         ]

#     def __getitem__(self, idx):
#         waveform, labels_str, conv_history = super().__getitem__(idx)
#         instruction_phrase = random.choice(self.instruction_phrases)

#         pre_speech_prompt = f"Instruction:\n{instruction_phrase} - ["
#         pre_speech_prompt += (
#             ", ".join(["IsSpeech" if k == "isSpeech" else k for k in labels_str.keys()])
#             + "]\n\nInput:\n<speech>"
#         )
#         pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
#         post_speech_prompt = f"</speech>\n\n" + "Output:\n"
#         output_prompt = "{"
#         for key, value in labels_str.items():
#             if key == "Isspeech":
#                 key = "SpeechActivity"
#             output_prompt += f'  "{key}": "{value}", '
#         output_prompt = output_prompt.rstrip(",\n") + "}"

#         complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
#         return (
#             waveform,
#             pre_speech_prompt,
#             post_speech_prompt,
#             output_prompt,
#             complete_prompt,
#         )


# # Example usage
# if __name__ == "__main__":
#     dataset = InstructionalAudioDataset(
#         csv_file="dev.csv", mode="test", random_keys_prob=0.0001
#     )
#     waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt = (
#         dataset[121]
#     )

#     print(complete_prompt)
#     print(waveform)
