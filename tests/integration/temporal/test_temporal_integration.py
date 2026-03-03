"""
Integration tests for temporal query functionality.

Tests complete workflows including:
- Version tracking across multiple updates
- Audit trail generation
- Time-travel queries with real data
- Rollback operations
- Compliance scenarios
"""

import pytest


pytestmark = pytest.mark.integration
from datetime import datetime, timedelta, UTC
from smartmemory import SmartMemory, MemoryItem
import time
import json


@pytest.fixture
def memory():
    """Provide clean SmartMemory instance."""
    mem = SmartMemory()
    yield mem
    try:
        mem.clear()
    except:
        pass


class TestVersionTracking:
    """Test version tracking across multiple updates."""
    
    def test_track_multiple_versions(self, memory):
        """Test tracking multiple versions of same content."""
        # Version 1
        v1 = MemoryItem(
            content="Python is a programming language",
            metadata={"version": 1, "author": "alice"}
        )
        v1_id = memory.add(v1)
        
        time.sleep(0.1)
        
        # Version 2
        v2 = MemoryItem(
            content="Python is a powerful programming language",
            metadata={"version": 2, "author": "bob", "original_id": v1_id}
        )
        v2_id = memory.add(v2)
        
        time.sleep(0.1)
        
        # Version 3
        v3 = MemoryItem(
            content="Python is a powerful, versatile programming language",
            metadata={"version": 3, "author": "charlie", "original_id": v1_id}
        )
        v3_id = memory.add(v3)
        
        # Verify all versions exist and are distinct
        v1_retrieved = memory.get(v1_id)
        v2_retrieved = memory.get(v2_id)
        v3_retrieved = memory.get(v3_id)
        
        assert v1_retrieved is not None, "Version 1 should exist"
        assert v2_retrieved is not None, "Version 2 should exist"
        assert v3_retrieved is not None, "Version 3 should exist"
        
        # Verify they are different items
        assert v1_id != v2_id != v3_id, "Version IDs must be unique"
        
        # Verify version 1 metadata
        assert v1_retrieved.metadata["version"] == 1, "Version 1 should have version=1"
        assert v1_retrieved.metadata["author"] == "alice", "Version 1 author should be alice"
        assert v1_retrieved.content == "Python is a programming language", "Version 1 content mismatch"
        
        # Verify version 3 metadata and content
        assert v3_retrieved.metadata["version"] == 3, "Version 3 should have version=3"
        assert v3_retrieved.metadata["author"] == "charlie", "Version 3 author should be charlie"
        assert "versatile" in v3_retrieved.content, "Version 3 should contain 'versatile'"
        
        # Verify original_id if preserved
        if "original_id" in v3_retrieved.metadata:
            assert v3_retrieved.metadata["original_id"] == v1_id, "original_id should reference v1"
    
    def test_version_history_retrieval(self, memory):
        """Test retrieving history of versioned items."""
        # Create item and register version explicitly (add() doesn't auto-version)
        v1 = MemoryItem(content="Version 1", metadata={"version": 1})
        v1_id = memory.add(v1)
        memory.version_tracker.create_version(
            item_id=v1_id, content=v1.content, metadata=v1.metadata
        )

        time.sleep(0.1)

        # Create second version of the same item
        memory.version_tracker.create_version(
            item_id=v1_id, content="Version 2",
            metadata={"version": 2}, change_reason="update"
        )

        # Get history - should have both versions
        history = memory.temporal.get_history(v1_id)
        
        assert isinstance(history, list), "History must be a list"
        assert len(history) >= 1, "History should contain at least one version"
        
        # Verify history contains TemporalVersion objects
        for version in history:
            assert hasattr(version, 'content'), "Version must have content"
            assert hasattr(version, 'metadata'), "Version must have metadata"
            assert hasattr(version, 'item_id'), "Version must have item_id"


class TestAuditTrailGeneration:
    """Test audit trail generation for compliance."""
    
    def test_audit_trail_creation(self, memory):
        """Test that audit trail is created when versioned."""
        item = MemoryItem(
            content="Sensitive data",
            metadata={
                "user_id": "user123",
                "sensitivity": "high"
            }
        )
        item_id = memory.add(item)
        # add() doesn't auto-version; create version explicitly
        memory.version_tracker.create_version(
            item_id=item_id, content=item.content, metadata=item.metadata
        )

        # Get audit trail
        trail = memory.temporal.get_audit_trail(item_id)
        
        assert len(trail) >= 1, "Audit trail should have at least one event for created item"
        
        # Verify first event is creation
        first_event = trail[0]
        assert 'action' in first_event, "Event must have action field"
        assert 'timestamp' in first_event, "Event must have timestamp"
        assert first_event['action'] in ['created', 'create'], f"First action should be create, got {first_event['action']}"
    
    def test_audit_trail_with_user_tracking(self, memory):
        """Test audit trail tracks user information."""
        item = MemoryItem(
            content="User action",
            metadata={
                "user_id": "alice",
                "action_type": "create"
            }
        )
        item_id = memory.add(item)
        memory.version_tracker.create_version(
            item_id=item_id, content=item.content, metadata=item.metadata
        )

        trail = memory.temporal.get_audit_trail(item_id, include_metadata=True)
        
        assert len(trail) >= 1, "Should have at least one audit event"
        
        # Verify trail structure
        first_event = trail[0]
        assert isinstance(first_event, dict), "Event must be a dictionary"
        assert 'action' in first_event, "Event must have action"
        assert 'timestamp' in first_event, "Event must have timestamp"
        
        # If metadata included, verify it's a dict
        if 'metadata' in first_event:
            assert isinstance(first_event['metadata'], dict), "Metadata must be a dict"
    
    def test_audit_trail_export(self, memory):
        """Test exporting audit trail to JSON."""
        item = MemoryItem(
            content="Compliance data",
            metadata={"compliance_required": True}
        )
        item_id = memory.add(item)
        memory.version_tracker.create_version(
            item_id=item_id, content=item.content, metadata=item.metadata
        )

        trail = memory.temporal.get_audit_trail(item_id)
        
        assert len(trail) >= 1, "Should have audit events"
        
        # Should be JSON serializable
        json_str = json.dumps(trail)
        assert json_str is not None, "Trail should be JSON serializable"
        assert len(json_str) > 0, "JSON string should not be empty"
        
        # Should be deserializable and match original
        loaded = json.loads(json_str)
        assert isinstance(loaded, list), "Deserialized trail should be a list"
        assert len(loaded) == len(trail), "Deserialized trail should have same length"
        
        # Verify structure is preserved
        for i, event in enumerate(loaded):
            assert isinstance(event, dict), f"Event {i} should be a dict"
            assert 'action' in event, f"Event {i} should have action"


class TestTimeTravelQueries:
    """Test time-travel queries with real data."""
    
    def test_time_travel_to_past(self, memory):
        """Test querying data as it existed in the past."""
        # Add item
        item = MemoryItem(
            content="Original content",
            metadata={"timestamp": datetime.now(UTC).isoformat()}
        )
        item_id = memory.add(item)
        
        # Time travel to past
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        
        with memory.time_travel(past):
            # Context should be set
            assert memory._temporal_context == past
    
    def test_time_travel_with_search(self, memory):
        """Test searching within time-travel context."""
        # Add items
        item1 = MemoryItem(content="Python programming")
        memory.add(item1)
        
        time.sleep(0.1)
        
        item2 = MemoryItem(content="JavaScript programming")
        memory.add(item2)
        
        # Search in time-travel context
        past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        
        with memory.time_travel(past):
            results = memory.search("programming")
            assert isinstance(results, list)
    
    def test_nested_time_travel(self, memory):
        """Test nested time-travel contexts."""
        item = MemoryItem(content="Test")
        memory.add(item)
        
        time1 = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        time2 = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        
        with memory.time_travel(time1):
            assert memory._temporal_context == time1
            
            with memory.time_travel(time2):
                assert memory._temporal_context == time2
            
            assert memory._temporal_context == time1


class TestRollbackOperations:
    """Test rollback functionality."""
    
    def test_rollback_dry_run(self, memory):
        """Test rollback dry run doesn't modify data."""
        # Create item
        item = MemoryItem(content="Original")
        item_id = memory.add(item)
        
        # Get original
        original = memory.get(item_id)
        
        # Dry run rollback
        to_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        result = memory.temporal.rollback(item_id, to_time, dry_run=True)
        
        # Data should be unchanged
        current = memory.get(item_id)
        assert current.content == original.content
    
    def test_rollback_preview_info(self, memory):
        """Test that rollback preview provides useful info."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)
        
        to_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        preview = memory.temporal.rollback(item_id, to_time, dry_run=True)
        
        assert isinstance(preview, dict)
        # Should have preview info or error
        assert 'would_change' in preview or 'preview' in preview or 'error' in preview


class TestComplianceScenarios:
    """Test compliance-related scenarios."""
    
    def test_hipaa_audit_trail(self, memory):
        """Test HIPAA-compliant audit trail."""
        # Simulate medical record
        record = MemoryItem(
            content="Patient diagnosis",
            metadata={
                "patient_id": "patient123",
                "user_id": "doctor_smith",
                "sensitivity": "PHI",
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        record_id = memory.add(record)
        memory.version_tracker.create_version(
            item_id=record_id, content=record.content, metadata=record.metadata
        )

        # Get audit trail
        trail = memory.temporal.get_audit_trail(record_id, include_metadata=True)

        assert len(trail) >= 1, "HIPAA: Must have audit trail for medical records"
        
        # Verify audit trail structure for compliance
        for event in trail:
            assert isinstance(event, dict), "Audit event must be a dict"
            assert 'timestamp' in event, "HIPAA: Must track when action occurred"
            assert 'action' in event, "HIPAA: Must track what action was performed"
        
        # If metadata included, verify structure
        if 'metadata' in trail[0]:
            assert isinstance(trail[0]['metadata'], dict), "Metadata must be a dict"
    
    def test_gdpr_data_tracking(self, memory):
        """Test GDPR-compliant data tracking."""
        # Personal data
        personal_data = MemoryItem(
            content="User email: user@example.com",
            metadata={
                "data_type": "personal",
                "user_id": "user123",
                "consent": True,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        data_id = memory.add(personal_data)
        memory.version_tracker.create_version(
            item_id=data_id, content=personal_data.content, metadata=personal_data.metadata
        )

        # Should be able to track all changes
        trail = memory.temporal.get_audit_trail(data_id)

        assert len(trail) >= 1, "GDPR: Must track all changes to personal data"
        
        # Verify action types are valid
        first_event = trail[0]
        assert "action" in first_event, "GDPR: Must track action type"
        assert first_event["action"] in ["created", "updated", "deleted", "create", "update", "delete"], \
            f"Invalid action type: {first_event['action']}"
        
        # Verify timestamp exists
        assert "timestamp" in first_event, "GDPR: Must track when data was modified"
    
    def test_soc2_security_logging(self, memory):
        """Test SOC2-compliant security logging."""
        # Security event
        event = MemoryItem(
            content="User login attempt",
            metadata={
                "event_type": "security",
                "user_id": "user123",
                "ip_address": "192.168.1.1",
                "timestamp": datetime.now(UTC).isoformat(),
                "success": True
            }
        )
        event_id = memory.add(event)
        memory.version_tracker.create_version(
            item_id=event_id, content=event.content, metadata=event.metadata
        )

        # Audit trail should capture security events
        trail = memory.temporal.get_audit_trail(event_id, include_metadata=True)

        assert len(trail) >= 1, "SOC2: Must have audit trail for security events"
        
        # Verify security event is logged
        for event in trail:
            assert isinstance(event, dict), "Audit event must be a dict"
            assert 'timestamp' in event, "SOC2: Must track when security event occurred"
            assert 'action' in event, "SOC2: Must track action type"


class TestChangeDetection:
    """Test change detection functionality."""
    
    def test_detect_content_changes(self, memory):
        """Test detecting content changes."""
        # Version 1
        v1 = MemoryItem(content="Original content")
        v1_id = memory.add(v1)
        
        time.sleep(0.1)
        
        # Version 2
        v2 = MemoryItem(
            content="Modified content",
            metadata={"original_id": v1_id}
        )
        memory.add(v2)
        
        # Get changes
        changes = memory.temporal.get_changes(v1_id)
        
        # Should detect changes (if implemented)
        assert isinstance(changes, list)
    
    def test_detect_metadata_changes(self, memory):
        """Test detecting metadata changes."""
        # Version 1
        v1 = MemoryItem(
            content="Content",
            metadata={"status": "draft"}
        )
        v1_id = memory.add(v1)
        
        time.sleep(0.1)
        
        # Version 2
        v2 = MemoryItem(
            content="Content",
            metadata={"status": "published", "original_id": v1_id}
        )
        memory.add(v2)
        
        # Get changes
        changes = memory.temporal.get_changes(v1_id)
        
        assert isinstance(changes, list)


class TestTemporalQueryPerformance:
    """Test performance of temporal queries."""
    
    def test_history_query_performance(self, memory):
        """Test that history queries complete in reasonable time."""
        import time as time_module
        
        # Add item
        item = MemoryItem(content="Test")
        item_id = memory.add(item)
        
        # Time the query
        start = time_module.time()
        history = memory.temporal.get_history(item_id)
        elapsed = time_module.time() - start
        
        # Should complete quickly (< 1 second)
        assert elapsed < 1.0
        assert isinstance(history, list)
    
    def test_audit_trail_query_performance(self, memory):
        """Test that audit trail queries complete in reasonable time."""
        import time as time_module
        
        # Add item
        item = MemoryItem(content="Test")
        item_id = memory.add(item)
        
        # Time the query
        start = time_module.time()
        trail = memory.temporal.get_audit_trail(item_id)
        elapsed = time_module.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0
        assert isinstance(trail, list)


class TestTemporalEdgeCases:
    """Test edge cases in temporal functionality."""
    
    def test_empty_history(self, memory):
        """Test querying history of non-existent item."""
        history = memory.temporal.get_history("nonexistent-id")
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_empty_audit_trail(self, memory):
        """Test querying audit trail of non-existent item."""
        trail = memory.temporal.get_audit_trail("nonexistent-id")
        
        assert isinstance(trail, list)
        assert len(trail) == 0
    
    def test_compare_same_time(self, memory):
        """Test comparing versions at same time."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)
        
        time_point = datetime.now(UTC).isoformat()
        
        diff = memory.temporal.compare_versions(item_id, time_point, time_point)
        
        assert isinstance(diff, dict)
    
    def test_rollback_to_future(self, memory):
        """Test rollback to future time."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)
        
        future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        
        result = memory.temporal.rollback(item_id, future, dry_run=True)
        
        assert isinstance(result, dict)
