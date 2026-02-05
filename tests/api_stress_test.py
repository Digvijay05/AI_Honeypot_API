#!/usr/bin/env python3
"""
API Stress Test Suite
=====================

A comprehensive stress-testing and validation suite for a REST API.
Uses raw curl commands via subprocess to simulate realistic client behavior.

This file tests:
- Correctness: Valid requests return expected responses
- Edge Cases: Malformed input, missing fields, boundary conditions
- Load: High-frequency bursts, concurrent sessions
- Fault Injection: Invalid auth, slow clients, oversized payloads

Author: Auto-generated for API validation
Python: 3.10+
Dependencies: None (stdlib only)
"""

import subprocess
import json
import time
import uuid
import random
import statistics
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURATION
# =============================================================================

# Deterministic seed for reproducibility
random.seed(42)

# API Configuration
BASE_URL: str = "https://askew-boreal-fidelia.ngrok-free.dev/"  # Replace with actual endpoint
API_KEY: str = "your-secure-api-key-here"               # Replace with actual API key
ENDPOINT_PATH: str = "/api/honeypot"             # Replace with actual path

# Timeout Configuration (seconds)
CURL_TIMEOUT: int = 30          # Max time for a single curl request
CURL_CONNECT_TIMEOUT: int = 10  # Max time for connection establishment

# Concurrency Configuration
MAX_CONCURRENT_SESSIONS: int = 10   # Number of parallel sessions for load tests
BURST_REQUEST_COUNT: int = 50       # Number of requests in burst test
BURST_DELAY_MS: int = 10            # Delay between burst requests (milliseconds)

# Payload Templates
def generate_session_id() -> str:
    """Generate a deterministic-looking session ID."""
    return f"test-session-{uuid.uuid4().hex[:12]}"

def make_initial_request_payload(session_id: str, message: str = "Hello, I am interested in your offer.") -> dict:
    """
    Template for a valid first-message request (empty conversationHistory).
    This tests the API's ability to handle new sessions.
    """
    return {
        "sessionId": session_id,
        "message": message,
        "conversationHistory": [],
        "timestamp": int(time.time() * 1000)  # Epoch milliseconds
    }

def make_followup_request_payload(session_id: str, history: list, message: str = "Tell me more about this.") -> dict:
    """
    Template for a follow-up request (non-empty conversationHistory).
    This tests stateful conversation handling.
    """
    return {
        "sessionId": session_id,
        "message": message,
        "conversationHistory": history,
        "timestamp": int(time.time() * 1000)
    }

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RequestResult:
    """Captures the outcome of a single API request."""
    test_name: str
    curl_command: str
    http_status: Optional[int]
    response_body: str
    latency_ms: float
    success: bool
    error_message: str = ""
    is_json_response: bool = True

@dataclass
class TestReport:
    """Aggregated metrics from all tests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    non_json_responses: int = 0
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)

# =============================================================================
# CURL EXECUTION ENGINE
# =============================================================================

def build_curl_command(
    method: str,
    url: str,
    headers: dict,
    data: Optional[str] = None,
    timeout: int = CURL_TIMEOUT,
    connect_timeout: int = CURL_CONNECT_TIMEOUT,
    extra_args: Optional[list] = None
) -> list:
    """
    Constructs a curl command as a list of arguments.
    
    We use -w to extract HTTP status code and timing info.
    We use -s for silent mode (no progress bar).
    We use -o /dev/null initially, then capture body separately.
    """
    cmd = [
        "curl",
        "-s",                                    # Silent mode
        "-X", method,                            # HTTP method
        "--connect-timeout", str(connect_timeout),
        "--max-time", str(timeout),
        "-w", "\n%{http_code}\n%{time_total}",  # Write out status and timing
    ]
    
    # Add headers
    for key, value in headers.items():
        cmd.extend(["-H", f"{key}: {value}"])
    
    # Add request body if present
    if data is not None:
        cmd.extend(["-d", data])
    
    # Add any extra arguments (for fault injection tests)
    if extra_args:
        cmd.extend(extra_args)
    
    # Add URL last
    cmd.append(url)
    
    return cmd

def execute_curl(cmd: list) -> tuple[Optional[int], str, float, str]:
    """
    Executes a curl command and parses the output.
    
    Returns:
        - HTTP status code (or None if failed)
        - Response body
        - Latency in milliseconds
        - Error message (empty if successful)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CURL_TIMEOUT + 5  # Extra buffer for subprocess
        )
        
        # Parse output: body lines, then status code, then time
        output_lines = result.stdout.strip().split("\n")
        
        if len(output_lines) < 2:
            # Malformed output, possibly a curl error
            return None, result.stdout, 0.0, result.stderr or "Malformed curl output"
        
        # Extract timing and status from last two lines
        time_total = output_lines[-1]
        http_status = output_lines[-2]
        body = "\n".join(output_lines[:-2])
        
        try:
            status_code = int(http_status)
            latency_ms = float(time_total) * 1000  # Convert to milliseconds
        except ValueError:
            return None, body, 0.0, f"Failed to parse status/timing: {http_status}, {time_total}"
        
        return status_code, body, latency_ms, ""
        
    except subprocess.TimeoutExpired:
        return None, "", 0.0, "Request timed out"
    except Exception as e:
        return None, "", 0.0, str(e)

def run_test(
    test_name: str,
    method: str,
    url: str,
    headers: dict,
    data: Optional[str] = None,
    extra_args: Optional[list] = None,
    expected_status: Optional[int] = None
) -> RequestResult:
    """
    Executes a single test case and returns the result.
    
    Args:
        test_name: Descriptive name for logging/reporting
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        headers: Dict of HTTP headers
        data: Optional request body (string, usually JSON)
        extra_args: Additional curl arguments for special tests
        expected_status: If set, success is determined by matching this status
    """
    cmd = build_curl_command(method, url, headers, data, extra_args=extra_args)
    cmd_str = " ".join(cmd)  # For logging
    
    status, body, latency, error = execute_curl(cmd)
    
    # Determine if response is valid JSON
    is_json = False
    if body:
        try:
            json.loads(body)
            is_json = True
        except json.JSONDecodeError:
            pass
    
    # Determine success
    if error:
        success = False
    elif expected_status is not None:
        success = (status == expected_status)
    else:
        # Default: 2xx is success
        success = status is not None and 200 <= status < 300
    
    return RequestResult(
        test_name=test_name,
        curl_command=cmd_str,
        http_status=status,
        response_body=body[:500] if body else "",  # Truncate for sanity
        latency_ms=latency,
        success=success,
        error_message=error,
        is_json_response=is_json
    )

# =============================================================================
# TEST SUITES
# =============================================================================

def get_full_url() -> str:
    """Constructs the full API URL."""
    return f"{BASE_URL.rstrip('/')}{ENDPOINT_PATH}"

def get_headers(api_key: str = API_KEY) -> dict:
    """Returns standard headers for API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key
    }

# -----------------------------------------------------------------------------
# CORRECTNESS TESTS
# These verify that the API handles valid inputs correctly.
# -----------------------------------------------------------------------------

def test_valid_initial_request(report: TestReport) -> None:
    """
    Test: Valid first message with empty conversation history.
    
    Why this matters: Establishes baseline functionality. If this fails,
    nothing else will work. Validates session initialization.
    """
    print("\n[CORRECTNESS] Testing valid initial request...")
    
    session_id = generate_session_id()
    payload = make_initial_request_payload(session_id)
    
    result = run_test(
        test_name="Valid Initial Request",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload)
    )
    
    _record_result(report, result)
    _print_result(result)

def test_valid_followup_request(report: TestReport) -> None:
    """
    Test: Valid follow-up with non-empty conversation history.
    
    Why this matters: Validates stateful behavior. The API must correctly
    incorporate prior context into its response.
    """
    print("\n[CORRECTNESS] Testing valid follow-up request...")
    
    session_id = generate_session_id()
    
    # First, establish a session
    initial_payload = make_initial_request_payload(session_id)
    initial_result = run_test(
        test_name="Followup Setup - Initial",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(initial_payload)
    )
    _record_result(report, initial_result)
    
    # Now send follow-up
    history = [
        {"role": "user", "content": "Hello, I am interested."},
        {"role": "assistant", "content": "Great! How can I help?"}
    ]
    followup_payload = make_followup_request_payload(session_id, history)
    
    result = run_test(
        test_name="Valid Follow-up Request",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(followup_payload)
    )
    
    _record_result(report, result)
    _print_result(result)

def test_multi_turn_conversation(report: TestReport) -> None:
    """
    Test: Execute a multi-turn conversation (5 exchanges).
    
    Why this matters: Validates conversation continuity and context retention.
    Detects issues with growing context sizes or state corruption.
    """
    print("\n[CORRECTNESS] Testing multi-turn conversation (5 turns)...")
    
    session_id = generate_session_id()
    history = []
    messages = [
        "Hello, I saw your ad.",
        "Yes, tell me more about the price.",
        "Is there any discount available?",
        "Can I pay in installments?",
        "What are the next steps?"
    ]
    
    for i, msg in enumerate(messages):
        payload = {
            "sessionId": session_id,
            "message": msg,
            "conversationHistory": history.copy(),
            "timestamp": int(time.time() * 1000)
        }
        
        result = run_test(
            test_name=f"Multi-turn Conversation - Turn {i+1}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
        
        _record_result(report, result)
        
        # Update history for next turn
        history.append({"role": "user", "content": msg})
        if result.is_json_response and result.response_body:
            try:
                resp = json.loads(result.response_body)
                if "response" in resp:
                    history.append({"role": "assistant", "content": resp["response"][:100]})
            except json.JSONDecodeError:
                pass
    
    print(f"  Completed 5-turn conversation for session {session_id}")

# -----------------------------------------------------------------------------
# EDGE CASE TESTS
# These verify that the API handles invalid/edge inputs gracefully.
# -----------------------------------------------------------------------------

def test_missing_session_id(report: TestReport) -> None:
    """
    Test: Request without sessionId field.
    
    Why this matters: The API should return a clear 400 error, not crash.
    Validates input validation layer.
    """
    print("\n[EDGE CASE] Testing missing sessionId...")
    
    payload = {
        "message": "Hello",
        "conversationHistory": [],
        "timestamp": int(time.time() * 1000)
    }
    
    result = run_test(
        test_name="Missing sessionId",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        expected_status=422  # Typically validation error
    )
    
    _record_result(report, result)
    _print_result(result)

def test_missing_message(report: TestReport) -> None:
    """
    Test: Request without message field.
    
    Why this matters: Message is the core input. API must reject gracefully.
    """
    print("\n[EDGE CASE] Testing missing message field...")
    
    payload = {
        "sessionId": generate_session_id(),
        "conversationHistory": [],
        "timestamp": int(time.time() * 1000)
    }
    
    result = run_test(
        test_name="Missing message",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        expected_status=422
    )
    
    _record_result(report, result)
    _print_result(result)

def test_malformed_json(report: TestReport) -> None:
    """
    Test: Send syntactically invalid JSON.
    
    Why this matters: API must not crash on parse errors. Should return 400.
    """
    print("\n[EDGE CASE] Testing malformed JSON...")
    
    malformed_data = '{"sessionId": "test", "message": "hello", broken}'
    
    result = run_test(
        test_name="Malformed JSON",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=malformed_data,
        expected_status=400
    )
    
    _record_result(report, result)
    _print_result(result)

def test_empty_body(report: TestReport) -> None:
    """
    Test: Send request with empty body.
    
    Why this matters: Edge case that can cause null-pointer-style errors.
    """
    print("\n[EDGE CASE] Testing empty request body...")
    
    result = run_test(
        test_name="Empty Body",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data="",
        expected_status=400
    )
    
    _record_result(report, result)
    _print_result(result)

def test_null_values(report: TestReport) -> None:
    """
    Test: Send JSON with null values for required fields.
    
    Why this matters: Null handling differs from missing fields. Both must be handled.
    """
    print("\n[EDGE CASE] Testing null field values...")
    
    payload = {
        "sessionId": None,
        "message": None,
        "conversationHistory": None,
        "timestamp": None
    }
    
    result = run_test(
        test_name="Null Field Values",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        expected_status=422
    )
    
    _record_result(report, result)
    _print_result(result)

def test_wrong_types(report: TestReport) -> None:
    """
    Test: Send wrong data types for fields (string where int expected, etc.).
    
    Why this matters: Type coercion bugs can cause subtle data corruption.
    """
    print("\n[EDGE CASE] Testing wrong field types...")
    
    payload = {
        "sessionId": 12345,                    # Should be string
        "message": ["array", "instead"],       # Should be string
        "conversationHistory": "not-an-array", # Should be array
        "timestamp": "not-a-number"            # Should be int
    }
    
    result = run_test(
        test_name="Wrong Field Types",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        expected_status=422
    )
    
    _record_result(report, result)
    _print_result(result)

def test_extremely_large_payload(report: TestReport) -> None:
    """
    Test: Send a payload with very large message content (~1MB).
    
    Why this matters: Detects buffer overflows, memory issues, and lack of
    size limits. Should be rejected gracefully or handled correctly.
    """
    print("\n[EDGE CASE] Testing extremely large payload (~1MB)...")
    
    # Generate ~1MB of text
    large_message = "A" * (1024 * 1024)
    
    payload = {
        "sessionId": generate_session_id(),
        "message": large_message,
        "conversationHistory": [],
        "timestamp": int(time.time() * 1000)
    }
    
    result = run_test(
        test_name="Large Payload (1MB)",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        expected_status=413  # Payload Too Large (or 400/422)
    )
    
    _record_result(report, result)
    _print_result(result)

def test_extremely_long_history(report: TestReport) -> None:
    """
    Test: Send conversation history with 100+ entries.
    
    Why this matters: Tests context window handling and memory usage.
    """
    print("\n[EDGE CASE] Testing extremely long conversation history (100 entries)...")
    
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i}"}
        for i in range(100)
    ]
    
    payload = {
        "sessionId": generate_session_id(),
        "message": "Final message with long history",
        "conversationHistory": history,
        "timestamp": int(time.time() * 1000)
    }
    
    result = run_test(
        test_name="Long History (100 entries)",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload)
    )
    
    _record_result(report, result)
    _print_result(result)

def test_special_characters(report: TestReport) -> None:
    """
    Test: Message with special characters, unicode, and escape sequences.
    
    Why this matters: Injection attacks, encoding issues, JSON parse errors.
    """
    print("\n[EDGE CASE] Testing special characters and unicode...")
    
    special_message = (
        'Test with "quotes" and \'apostrophes\' '
        'and unicode: 你好 🎉 '
        'and escapes: \\n \\t \\\\ '
        'and HTML: <script>alert("xss")</script>'
    )
    
    payload = {
        "sessionId": generate_session_id(),
        "message": special_message,
        "conversationHistory": [],
        "timestamp": int(time.time() * 1000)
    }
    
    result = run_test(
        test_name="Special Characters",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload)
    )
    
    _record_result(report, result)
    _print_result(result)

# -----------------------------------------------------------------------------
# AUTHENTICATION TESTS
# These verify that the API properly enforces authentication.
# -----------------------------------------------------------------------------

def test_missing_api_key(report: TestReport) -> None:
    """
    Test: Request without x-api-key header.
    
    Why this matters: Must return 401/403, not proceed unauthenticated.
    """
    print("\n[AUTH] Testing missing API key...")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
        # No x-api-key
    }
    
    payload = make_initial_request_payload(generate_session_id())
    
    result = run_test(
        test_name="Missing API Key",
        method="POST",
        url=get_full_url(),
        headers=headers,
        data=json.dumps(payload),
        expected_status=401
    )
    
    _record_result(report, result)
    _print_result(result)

def test_invalid_api_key(report: TestReport) -> None:
    """
    Test: Request with incorrect API key.
    
    Why this matters: Ensures key validation is actually happening.
    """
    print("\n[AUTH] Testing invalid API key...")
    
    headers = get_headers(api_key="invalid-key-12345")
    payload = make_initial_request_payload(generate_session_id())
    
    result = run_test(
        test_name="Invalid API Key",
        method="POST",
        url=get_full_url(),
        headers=headers,
        data=json.dumps(payload),
        expected_status=403
    )
    
    _record_result(report, result)
    _print_result(result)

def test_empty_api_key(report: TestReport) -> None:
    """
    Test: Request with empty API key header.
    
    Why this matters: Edge case where header exists but value is empty.
    """
    print("\n[AUTH] Testing empty API key...")
    
    headers = get_headers(api_key="")
    payload = make_initial_request_payload(generate_session_id())
    
    result = run_test(
        test_name="Empty API Key",
        method="POST",
        url=get_full_url(),
        headers=headers,
        data=json.dumps(payload),
        expected_status=401
    )
    
    _record_result(report, result)
    _print_result(result)

# -----------------------------------------------------------------------------
# LOAD TESTS
# These verify the API's performance under stress.
# -----------------------------------------------------------------------------

def test_rapid_same_session(report: TestReport) -> None:
    """
    Test: Rapid repeated requests with the same sessionId.
    
    Why this matters: Detects race conditions in session state management.
    If the API uses locking, this can reveal deadlocks or starvation.
    """
    print(f"\n[LOAD] Testing rapid requests (same session, {BURST_REQUEST_COUNT} requests)...")
    
    session_id = generate_session_id()
    results = []
    
    for i in range(BURST_REQUEST_COUNT):
        payload = make_initial_request_payload(session_id, f"Rapid message {i}")
        
        result = run_test(
            test_name=f"Rapid Same Session - {i}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
        
        results.append(result)
        _record_result(report, result)
        
        time.sleep(BURST_DELAY_MS / 1000)  # Small delay between requests
    
    success_count = sum(1 for r in results if r.success)
    print(f"  Completed: {success_count}/{BURST_REQUEST_COUNT} successful")
    
    # Check for increasing latency (memory leak symptom)
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if len(latencies) >= 10:
        first_half_avg = statistics.mean(latencies[:len(latencies)//2])
        second_half_avg = statistics.mean(latencies[len(latencies)//2:])
        if second_half_avg > first_half_avg * 1.5:
            print(f"  WARNING: Latency increased significantly ({first_half_avg:.0f}ms -> {second_half_avg:.0f}ms)")

def test_concurrent_sessions(report: TestReport) -> None:
    """
    Test: Concurrent requests with different sessionIds.
    
    Why this matters: Validates thread safety and isolation between sessions.
    Detects shared state corruption.
    """
    print(f"\n[LOAD] Testing concurrent sessions ({MAX_CONCURRENT_SESSIONS} parallel)...")
    
    results = []
    
    def make_request(session_num: int) -> RequestResult:
        session_id = f"concurrent-session-{session_num}"
        payload = make_initial_request_payload(session_id)
        return run_test(
            test_name=f"Concurrent Session - {session_num}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SESSIONS) as executor:
        futures = [executor.submit(make_request, i) for i in range(MAX_CONCURRENT_SESSIONS)]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _record_result(report, result)
    
    success_count = sum(1 for r in results if r.success)
    print(f"  Completed: {success_count}/{MAX_CONCURRENT_SESSIONS} successful")

def test_burst_traffic(report: TestReport) -> None:
    """
    Test: High-frequency burst traffic (as fast as possible).
    
    Why this matters: Simulates traffic spikes. Tests rate limiting,
    connection pooling, and resource exhaustion handling.
    """
    print(f"\n[LOAD] Testing burst traffic ({BURST_REQUEST_COUNT} requests, no delay)...")
    
    results = []
    start_time = time.time()
    
    def make_burst_request(i: int) -> RequestResult:
        session_id = f"burst-{i}"
        payload = make_initial_request_payload(session_id)
        return run_test(
            test_name=f"Burst - {i}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_burst_request, i) for i in range(BURST_REQUEST_COUNT)]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _record_result(report, result)
    
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.success)
    throughput = len(results) / elapsed if elapsed > 0 else 0
    
    print(f"  Completed: {success_count}/{BURST_REQUEST_COUNT} successful")
    print(f"  Throughput: {throughput:.1f} requests/second")

# -----------------------------------------------------------------------------
# FAULT INJECTION TESTS
# These simulate failure conditions.
# -----------------------------------------------------------------------------

def test_slow_client(report: TestReport) -> None:
    """
    Test: Simulate a slow client by sending data in chunks (via curl options).
    
    Why this matters: Tests server timeout handling for slow uploads.
    Detects resource leaks from hanging connections.
    """
    print("\n[FAULT] Testing slow client simulation...")
    
    payload = make_initial_request_payload(generate_session_id())
    
    # Use curl's --limit-rate to simulate slow upload
    result = run_test(
        test_name="Slow Client",
        method="POST",
        url=get_full_url(),
        headers=get_headers(),
        data=json.dumps(payload),
        extra_args=["--limit-rate", "1k"]  # 1KB/s upload rate
    )
    
    _record_result(report, result)
    _print_result(result)

def test_connection_reuse(report: TestReport) -> None:
    """
    Test: Verify behavior with keepalive connections.
    
    Why this matters: Tests connection pooling and reuse logic.
    """
    print("\n[FAULT] Testing connection reuse pattern...")
    
    session_id = generate_session_id()
    
    # Send multiple requests that could reuse connections
    for i in range(5):
        payload = make_initial_request_payload(session_id, f"Keepalive test {i}")
        result = run_test(
            test_name=f"Connection Reuse - {i}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
        _record_result(report, result)
    
    print("  Completed 5 sequential requests")

# -----------------------------------------------------------------------------
# IDEMPOTENCY TESTS
# These verify consistent behavior for repeated identical requests.
# -----------------------------------------------------------------------------

def test_idempotency(report: TestReport) -> None:
    """
    Test: Send identical requests multiple times.
    
    Why this matters: Responses should be consistent (or predictably different).
    Detects race conditions and non-deterministic behavior.
    """
    print("\n[IDEMPOTENCY] Testing response consistency...")
    
    session_id = generate_session_id()
    payload = make_initial_request_payload(session_id, "Idempotency test message")
    
    responses = []
    for i in range(3):
        result = run_test(
            test_name=f"Idempotency - {i}",
            method="POST",
            url=get_full_url(),
            headers=get_headers(),
            data=json.dumps(payload)
        )
        responses.append(result)
        _record_result(report, result)
        time.sleep(0.5)  # Brief pause between requests
    
    # Check consistency (status codes should match)
    statuses = [r.http_status for r in responses]
    if len(set(statuses)) > 1:
        print(f"  WARNING: Inconsistent status codes: {statuses}")
    else:
        print(f"  Consistent: All requests returned status {statuses[0]}")

# =============================================================================
# REPORTING
# =============================================================================

def _record_result(report: TestReport, result: RequestResult) -> None:
    """Records a single result into the aggregate report."""
    report.total_requests += 1
    
    if result.success:
        report.successful_requests += 1
    else:
        report.failed_requests += 1
        report.errors.append((result.test_name, result.error_message or f"Status: {result.http_status}"))
    
    if result.error_message and "timed out" in result.error_message.lower():
        report.timeouts += 1
    
    if not result.is_json_response and result.response_body:
        report.non_json_responses += 1
    
    if result.latency_ms > 0:
        report.latencies_ms.append(result.latency_ms)

def _print_result(result: RequestResult) -> None:
    """Prints a single test result."""
    status = "✓" if result.success else "✗"
    print(f"  {status} [{result.test_name}] Status: {result.http_status}, Latency: {result.latency_ms:.0f}ms")
    if result.error_message:
        print(f"    Error: {result.error_message}")

def print_final_report(report: TestReport) -> None:
    """Prints the aggregated test report."""
    print("\n" + "=" * 70)
    print("FINAL TEST REPORT")
    print("=" * 70)
    
    print(f"\n📊 REQUEST SUMMARY")
    print(f"   Total Requests:      {report.total_requests}")
    print(f"   Successful:          {report.successful_requests}")
    print(f"   Failed:              {report.failed_requests}")
    print(f"   Success Rate:        {(report.successful_requests / report.total_requests * 100):.1f}%" if report.total_requests > 0 else "N/A")
    
    print(f"\n⚠️  ERROR SUMMARY")
    print(f"   Timeouts:            {report.timeouts}")
    print(f"   Non-JSON Responses:  {report.non_json_responses}")
    
    if report.latencies_ms:
        print(f"\n⏱️  LATENCY SUMMARY (ms)")
        print(f"   Min:                 {min(report.latencies_ms):.0f}")
        print(f"   Max:                 {max(report.latencies_ms):.0f}")
        print(f"   Average:             {statistics.mean(report.latencies_ms):.0f}")
        print(f"   Median:              {statistics.median(report.latencies_ms):.0f}")
        
        if len(report.latencies_ms) >= 20:
            sorted_latencies = sorted(report.latencies_ms)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            print(f"   P95:                 {sorted_latencies[p95_idx]:.0f}")
            print(f"   P99:                 {sorted_latencies[p99_idx]:.0f}")
    
    if report.errors:
        print(f"\n❌ ERROR DETAILS (first 10)")
        for test_name, error in report.errors[:10]:
            print(f"   - {test_name}: {error[:80]}")
    
    print("\n" + "=" * 70)
    
    # Exit code recommendation
    if report.failed_requests == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {report.failed_requests} test(s) failed. Review errors above.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tests() -> None:
    """Executes the complete test suite."""
    report = TestReport()
    
    print("=" * 70)
    print("API STRESS TEST SUITE")
    print("=" * 70)
    print(f"Target: {get_full_url()}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Correctness Tests
    test_valid_initial_request(report)
    test_valid_followup_request(report)
    test_multi_turn_conversation(report)
    
    # Edge Case Tests
    test_missing_session_id(report)
    test_missing_message(report)
    test_malformed_json(report)
    test_empty_body(report)
    test_null_values(report)
    test_wrong_types(report)
    test_extremely_large_payload(report)
    test_extremely_long_history(report)
    test_special_characters(report)
    
    # Authentication Tests
    test_missing_api_key(report)
    test_invalid_api_key(report)
    test_empty_api_key(report)
    
    # Load Tests
    test_rapid_same_session(report)
    test_concurrent_sessions(report)
    test_burst_traffic(report)
    
    # Fault Injection Tests
    test_slow_client(report)
    test_connection_reuse(report)
    
    # Idempotency Tests
    test_idempotency(report)
    
    # Final Report
    print_final_report(report)

if __name__ == "__main__":
    run_all_tests()
