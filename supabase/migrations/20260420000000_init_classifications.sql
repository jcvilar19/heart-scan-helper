-- Storage bucket for X-ray uploads (public for simplicity of signed previews)
insert into storage.buckets (id, name, public)
values ('xray-uploads', 'xray-uploads', true)
on conflict (id) do nothing;

-- Public read of the bucket
create policy "Public read xray-uploads"
on storage.objects for select
using (bucket_id = 'xray-uploads');

-- Authenticated users can upload to their own folder (folder name = uid)
create policy "Users upload own folder xray-uploads"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'xray-uploads'
  and (storage.foldername(name))[1] = auth.uid()::text
);

create policy "Users delete own folder xray-uploads"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'xray-uploads'
  and (storage.foldername(name))[1] = auth.uid()::text
);

-- Classification history
create table public.classifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  image_path text not null,
  image_name text not null,
  probability double precision not null check (probability >= 0 and probability <= 1),
  prediction smallint not null check (prediction in (0, 1)),
  pathology text not null default 'cardiomegaly',
  created_at timestamptz not null default now()
);

create index classifications_user_created_idx
  on public.classifications (user_id, created_at desc);

alter table public.classifications enable row level security;

create policy "Users read own classifications"
on public.classifications for select
to authenticated
using (auth.uid() = user_id);

create policy "Users insert own classifications"
on public.classifications for insert
to authenticated
with check (auth.uid() = user_id);

create policy "Users delete own classifications"
on public.classifications for delete
to authenticated
using (auth.uid() = user_id);
