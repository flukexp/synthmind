from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """Base agent class that all specific agents should inherit from."""
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process the input data and return a result.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Dict containing the processing result
        """
        pass
    
    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate the input data.
        
        Args:
            input_data: The input data to validate
            
        Returns:
            True if the input is valid, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True
    
    def _format_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        Format the raw response into a standardized output.
        
        Args:
            raw_response: The raw response from the underlying service
            
        Returns:
            Dict containing the formatted response
        """
        # Default implementation - can be overridden by subclasses
        if isinstance(raw_response, dict):
            return raw_response
        elif isinstance(raw_response, BaseModel):
            return raw_response.dict()
        else:
            return {"result": str(raw_response)}
    
    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors that occur during processing.
        
        Args:
            error: The exception that was raised
            
        Returns:
            Dict containing error information
        """
        return {
            "error": True,
            "message": str(error),
            "type": error.__class__.__name__
        }