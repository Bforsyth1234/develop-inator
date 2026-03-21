create extension if not exists vector;

create table if not exists public.slack_thread_messages (
    id bigint generated always as identity primary key,
    workspace_id text not null,
    channel_id text not null,
    thread_ts text not null,
    message_ts text not null,
    user_id text,
    username text,
    text text not null,
    payload jsonb not null default '{}'::jsonb,
    posted_at timestamptz,
    inserted_at timestamptz not null default timezone('utc', now()),
    unique (channel_id, message_ts)
);

create index if not exists slack_thread_messages_thread_lookup_idx
    on public.slack_thread_messages (channel_id, thread_ts, posted_at, message_ts);

create table if not exists public.documentation_chunks (
    id bigint generated always as identity primary key,
    source_type text not null,
    source_id text not null,
    chunk_index integer not null,
    title text,
    path text,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536) not null,
    inserted_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    unique (source_type, source_id, chunk_index)
);

create index if not exists documentation_chunks_source_lookup_idx
    on public.documentation_chunks (source_type, source_id, chunk_index);

create index if not exists documentation_chunks_metadata_gin_idx
    on public.documentation_chunks using gin (metadata);

create index if not exists documentation_chunks_embedding_cosine_idx
    on public.documentation_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create or replace function public.match_documentation_chunks(
    query_embedding vector(1536),
    match_count integer default 5,
    min_similarity double precision default 0.0,
    filter jsonb default '{}'::jsonb
)
returns table (
    source_type text,
    source_id text,
    chunk_index integer,
    title text,
    path text,
    content text,
    metadata jsonb,
    similarity double precision
)
language sql
stable
as $$
    select
        documentation_chunks.source_type,
        documentation_chunks.source_id,
        documentation_chunks.chunk_index,
        documentation_chunks.title,
        documentation_chunks.path,
        documentation_chunks.content,
        documentation_chunks.metadata,
        1 - (documentation_chunks.embedding <=> query_embedding) as similarity
    from public.documentation_chunks
    where documentation_chunks.metadata @> filter
      and 1 - (documentation_chunks.embedding <=> query_embedding) >= min_similarity
    order by documentation_chunks.embedding <=> query_embedding
    limit greatest(match_count, 1);
$$;