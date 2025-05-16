import torch

class Settings:
    def __init__(self):
        self.algo = 'Discop'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.length = 200
        self.seed = 42
        self.task = 'text'

# Default settings for different tasks
text_default_settings = Settings()

class ImageSettings(Settings):
    def __init__(self):
        super().__init__()
        self.task = 'image'

class AudioSettings(Settings):
    def __init__(self):
        super().__init__()
        self.task = 'audio'

image_default_settings = ImageSettings()
audio_default_settings = AudioSettings() 