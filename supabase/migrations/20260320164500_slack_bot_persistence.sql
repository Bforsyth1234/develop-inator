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

-- GIN index for full-text search on the content column.
create index if not exists documentation_chunks_content_fts_idx
    on public.documentation_chunks
    using gin (to_tsvector('english', content));

create or replace function public.match_documentation_chunks(
    query_embedding vector(1536),
    match_count integer default 20,
    min_similarity double precision default 0.0,
    filter jsonb default '{}'::jsonb,
    query_text text default ''
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
    -- Hybrid search: combine pgvector cosine similarity with PostgreSQL FTS
    -- using Reciprocal Rank Fusion (RRF).  The constant k=60 is the standard
    -- RRF smoothing parameter.  When query_text is empty the FTS component
    -- contributes nothing and the ordering falls back to pure semantic search.
    with semantic as (
        select
            dc.id,
            1 - (dc.embedding <=> query_embedding) as sem_score,
            row_number() over (order by dc.embedding <=> query_embedding) as sem_rank
        from public.documentation_chunks dc
        where dc.metadata @> filter
          and 1 - (dc.embedding <=> query_embedding) >= min_similarity
        order by dc.embedding <=> query_embedding
        limit greatest(match_count, 1) * 2  -- over-fetch for fusion
    ),
    keyword as (
        select
            dc.id,
            ts_rank_cd(to_tsvector('english', dc.content),
                        websearch_to_tsquery('english', query_text)) as kw_score,
            row_number() over (
                order by ts_rank_cd(to_tsvector('english', dc.content),
                                     websearch_to_tsquery('english', query_text)) desc
            ) as kw_rank
        from public.documentation_chunks dc
        where dc.metadata @> filter
          and query_text <> ''
          and to_tsvector('english', dc.content) @@ websearch_to_tsquery('english', query_text)
        order by kw_score desc
        limit greatest(match_count, 1) * 2
    ),
    fused as (
        select
            coalesce(s.id, k.id) as id,
            -- RRF score: 1/(k+rank) for each list, summed
            coalesce(1.0 / (60 + s.sem_rank), 0.0)
              + coalesce(1.0 / (60 + k.kw_rank), 0.0) as rrf_score,
            coalesce(s.sem_score, 0.0) as sem_score
        from semantic s
        full outer join keyword k on s.id = k.id
    )
    select
        dc.source_type,
        dc.source_id,
        dc.chunk_index,
        dc.title,
        dc.path,
        dc.content,
        dc.metadata,
        f.sem_score as similarity
    from fused f
    join public.documentation_chunks dc on dc.id = f.id
    order by f.rrf_score desc
    limit greatest(match_count, 1);
$$;