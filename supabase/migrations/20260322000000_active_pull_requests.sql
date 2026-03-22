create table if not exists public.active_pull_requests (
    pr_url text primary key,
    branch_name text not null,
    channel_id text not null,
    thread_ts text not null,
    status text not null default 'open',
    inserted_at timestamptz not null default timezone('utc', now())
);

create index if not exists active_pull_requests_status_idx
    on public.active_pull_requests (status);

create table if not exists public.action_executions (
    id text primary key,
    channel text not null,
    thread_ts text not null,
    user_id text,
    original_request text not null,
    generated_spec text not null,
    status text not null default 'pending',
    model text,
    inserted_at timestamptz not null default timezone('utc', now())
);

create index if not exists action_executions_thread_idx
    on public.action_executions (channel, thread_ts, status);

