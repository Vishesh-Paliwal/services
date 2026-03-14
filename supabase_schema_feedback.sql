-- ============================================================
-- AVIRA Feedback Tables
-- Run this in the Supabase SQL Editor
-- ============================================================

-- 1. Per-response feedback
create table if not exists response_feedback (
    id          uuid primary key default gen_random_uuid(),
    tester_id   uuid not null references auth.users(id) on delete cascade,
    session_id  text not null,
    response_number int not null,
    tester_role text,
    question_topic text,
    difficulty_tier text not null,

    q1_relevance          text not null,
    q2_scientific_accuracy text not null,
    q3_depth              text not null,
    q4_length_format      text not null,
    q5_critical_content   text not null,
    q7_comment            text not null,
    quick_score           int not null check (quick_score between 1 and 10),

    conversation_id uuid references conversations(id) on delete set null,
    message_id      uuid references messages(id) on delete set null,

    created_at timestamptz default now()
);

-- 2. Pattern checks (every 5/10/15 responses)
create table if not exists pattern_checks (
    id          uuid primary key default gen_random_uuid(),
    tester_id   uuid not null references auth.users(id) on delete cascade,
    session_id  text not null,
    checkpoint  int not null,

    strengths               text[] default '{}',
    weaknesses              text[] default '{}',
    confidence_vs_accuracy  text not null,
    comparison_to_usual_tool text not null,
    comparison_detail       text,

    created_at timestamptz default now()
);

-- Indexes
create index if not exists idx_response_feedback_session
    on response_feedback(session_id, tester_id);

create index if not exists idx_response_feedback_tester
    on response_feedback(tester_id);

create index if not exists idx_pattern_checks_session
    on pattern_checks(session_id, tester_id);

-- Row-Level Security
alter table response_feedback enable row level security;
alter table pattern_checks enable row level security;

-- Testers can insert and read their own feedback
create policy "Users can insert own feedback"
    on response_feedback for insert
    with check (tester_id = auth.uid());

create policy "Users can read own feedback"
    on response_feedback for select
    using (tester_id = auth.uid());

create policy "Users can insert own pattern checks"
    on pattern_checks for insert
    with check (tester_id = auth.uid());

create policy "Users can read own pattern checks"
    on pattern_checks for select
    using (tester_id = auth.uid());
