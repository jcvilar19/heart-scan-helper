CREATE POLICY "Users update own classifications"
ON public.classifications
FOR UPDATE
TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);