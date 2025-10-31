import os
import datetime

class Log:
    def __init__(self, directory="/tmp/", filename='ai.log'):
        """
        Initialize the logger with a specified directory and filename.
        Ensures the directory exists and opens a file for writing logs.
        """
        self.directory = directory
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename)

        # Create the log directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

        # Open the log file for appending
        self.file = open(self.filepath, 'a', encoding='utf-8')

    def info(self, message):
        """Log an informational message."""
        self._write_log('INFO', message)

    def warning(self, message):
        """Log a warning message."""
        self._write_log('WARNING', message)

    def error(self, message):
        """Log an error message."""
        self._write_log('ERROR', message)

    def _write_log(self, level, message):
        """
        Helper method to write a log message to the file.
        Formats the message with a timestamp and log level.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] [{level}] {message}\n'
        self.file.write(log_entry)
        self.file.flush()

    def close(self):
        """Close the log file to release resources."""
        self.file.close()