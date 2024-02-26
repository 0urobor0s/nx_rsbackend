# mix run bench/nx.exs

Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

# sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
sizes = [128, 256, 512, 1024]

functions = [
  &Nx.LinAlg.qr/1,
  &NxRSBackend.LinAlg.qr_binary/1,
  &NxRSBackend.LinAlg.qr_tensor/1
]

chunks = length(functions)

for size <- sizes, function <- functions do
  tensor =
    Nx.Random.key(42)
    |> Nx.Random.normal(shape: {size, size}, type: :f64)
    |> elem(0)

  func = Function.info(function)[:name]
  module = Function.info(function)[:module]

  ["#{module}_#{func}_#{size}": fn -> function.(tensor) end]
end
|> List.flatten()
|> Enum.chunk_every(chunks)
|> Enum.map(fn e -> Enum.into(e, %{}) end)
|> IO.inspect(label: "Run:")
|> Enum.map(&Benchee.run/1)
