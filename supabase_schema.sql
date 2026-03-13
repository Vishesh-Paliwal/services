-- ============================================
-- Supabase Schema: Auth + Chat History
-- Run this in Supabase SQL Editor
-- ============================================

-- 1. Tables

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  store_name text not null,
  title text not null default 'New conversation',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  metadata jsonb default '{}',
  created_at timestamptz default now()
);

-- 2. Indexes

create index idx_conversations_user_store on public.conversations(user_id, store_name);
create index idx_messages_conversation on public.messages(conversation_id, created_at);

-- 3. Row Level Security

alter table public.profiles enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

create policy "own_profile_select" on public.profiles for select using (auth.uid() = id);
create policy "own_profile_update" on public.profiles for update using (auth.uid() = id);

create policy "own_convos_select" on public.conversations for select using (auth.uid() = user_id);
create policy "own_convos_insert" on public.conversations for insert with check (auth.uid() = user_id);
create policy "own_convos_update" on public.conversations for update using (auth.uid() = user_id);
create policy "own_convos_delete" on public.conversations for delete using (auth.uid() = user_id);

create policy "own_msgs_select" on public.messages for select
  using (exists (select 1 from public.conversations where id = conversation_id and user_id = auth.uid()));
create policy "own_msgs_insert" on public.messages for insert
  with check (exists (select 1 from public.conversations where id = conversation_id and user_id = auth.uid()));

-- 4. Auto-create profile on signup

create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, new.raw_user_meta_data->>'display_name');
  return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
