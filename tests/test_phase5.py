"""
Phase 5 tests: The CLI loop.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from studyflow.agents.schemas import NotesResult, QuizResult, TaskStatus
from studyflow.cli import _get_downloads_dir


def test_get_downloads_dir():
    """Verify that downloads dir resolves to a Path."""
    downloads = _get_downloads_dir()
    assert isinstance(downloads, Path)
    assert downloads.name == "Downloads"


@pytest.mark.asyncio
async def test_handle_learn_writes_to_downloads():
    """Test that handle_learn invokes graph and writes notes to downloads."""
    from studyflow.cli import handle_learn
    from unittest.mock import mock_open as unittest_mock_open

    topic = "Test Topic"
    
    mock_final_state = {
        "topic": topic,
        "prerequisite_warnings": [],
        "errors": [],
        "notes_result": NotesResult(
            task_id="00000000-0000-0000-0000-000000000000",
            topic=topic,
            status=TaskStatus.COMPLETED,
            synthesized_notes="# Mock Notes for Download",
            prior_context_used=[],
            chunk_ids_stored=[]
        ),
        "quiz_result": QuizResult(
            task_id="00000000-0000-0000-0000-000000000000",
            topic=topic,
            status=TaskStatus.COMPLETED,
            questions=[]
        )
    }

    # We need to mock astream to yield a final step
    async def mock_astream(*args, **kwargs):
        yield {"final_node": mock_final_state}
        
    mock_graph = AsyncMock()
    mock_graph.astream = mock_astream

    # Mock open() to avoid actually writing to the filesystem
    m_open = unittest_mock_open()
    with patch("studyflow.cli.build_graph", return_value=mock_graph), \
         patch("builtins.open", m_open):
        
        await handle_learn(topic)
        
        # Verify file was written
        m_open.assert_called_once()
        args, kwargs = m_open.call_args
        file_path = str(args[0])
        assert "Downloads" in file_path
        assert "Test_Topic" in file_path
        
        # Verify content written
        m_open().write.assert_called_once_with("# Mock Notes for Download")


@pytest.mark.asyncio
async def test_handle_digest_writes_to_downloads():
    """Test that digest queries DB, generates, and writes to downloads."""
    from studyflow.cli import handle_digest
    from unittest.mock import mock_open as unittest_mock_open

    mock_provider = MagicMock()
    mock_provider.generate.return_value = "# Weekly Digest\nGreat job!"

    # Mock DB cursor and file system
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []

    m_open = unittest_mock_open()
    with patch("studyflow.cli._conn", mock_conn), \
         patch("studyflow.cli._repo"), \
         patch("studyflow.cli.get_provider", return_value=mock_provider), \
         patch("builtins.open", m_open):

        await handle_digest()
        
        mock_provider.generate.assert_called_once()
        
        m_open.assert_called_once()
        file_path = str(m_open.call_args[0][0])
        assert "Downloads" in file_path
        assert "WeeklyDigest" in file_path
        
        m_open().write.assert_called_once_with("# Weekly Digest\nGreat job!")
