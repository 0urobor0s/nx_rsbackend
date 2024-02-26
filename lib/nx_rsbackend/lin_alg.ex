defmodule NxRSBackend.LinAlg do
  alias NxRSBackend.RS

  # Only support f64
  def qr_binary(tensor) do
    {:f, 64} = Nx.type(tensor)
    {nrow, ncol} = Nx.shape(tensor)

    tensor
    |> Nx.to_binary()
    |> RS.qr_binary(nrow, ncol)
    |> handle_qr(nrow, ncol)
  end

  defp handle_qr({q, r}, nrow, ncol) do
    q = Nx.from_binary(q, :f64) |> Nx.reshape({nrow, ncol})
    r = Nx.from_binary(r, :f64) |> Nx.reshape({ncol, ncol})
    {q, r}
  end

  def qr_tensor(tensor) do
    {:f, 64} = Nx.type(tensor)
    {_nrow, _ncol} = Nx.shape(tensor)

    tensor
    |> convert_to_binary_backend()
    |> RS.qr_tensor()
    |> convert_to_given_backend(tensor.data.__struct__)
  end

  defp convert_to_given_backend(tensors, backend) when is_tuple(tensors) do
    tensors
    |> Tuple.to_list()
    |> Enum.map(fn t -> convert_to_given_backend(t, backend) end)
    |> List.to_tuple()
  end

  defp convert_to_given_backend(tensor, backend) do
    case backend do
      Nx.BinaryBackend ->
        tensor

      _ ->
        tensor
        |> Nx.to_binary()
        |> Nx.from_binary(:f64, backend: {backend, []})
        |> Nx.reshape(Nx.shape(tensor))
    end
  end

  defp convert_to_binary_backend(tensor) do
    case tensor.data.__struct__ do
      Nx.BinaryBackend ->
        tensor

      _ ->
        tensor
        |> Nx.to_binary()
        |> Nx.from_binary(:f64, backend: {Nx.BinaryBackend, []})
        |> Nx.reshape(Nx.shape(tensor))
    end
  end
end
